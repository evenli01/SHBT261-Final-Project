from .base_model import BaseModel
from transformers import AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re

class QwenModel(BaseModel):
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, device)
        self.load_model()

    def load_model(self):
        print(f"Loading Qwen model from {self.model_path}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        self.model.to(self.device)
        # Use Qwen2VLProcessor directly to avoid AutoProcessor bug
        from transformers import Qwen2VLProcessor
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)

    @staticmethod
    def _cleanup_answer(text: str, max_words: int = 6) -> str:
        """
        Heuristic cleanup to keep answers short and comparable to VQA annotations.
        Removes code fences/prefixes and truncates to a few words.
        """
        text = text.replace("```", " ").replace("\n", " ").strip()
        lower = text.lower()
        prefixes = [
            "the answer is", "answer is", "answer:", "the text in the image reads",
            "the text reads", "it reads", "it says", "text:",
        ]
        for p in prefixes:
            if lower.startswith(p):
                text = text[len(p):].strip(" :,-")
                break
        # Take the first clause
        text = re.split(r"[.;]", text)[0].strip()
        words = text.split()
        if max_words and len(words) > max_words:
            text = " ".join(words[:max_words])
        return text.strip()

    def generate_answer(self, image: Image.Image, question: str, max_new_tokens=15):
        # Encourage concise answers via a system prompt
        messages = [
            {
                "role": "system",
                "content": "You are answering visual questions. Respond with a short phrase, not a full sentence.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        # Use greedy decoding (most stable, no sampling randomness that can cause NaN)
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding - most stable, no probability sampling
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw_answer = output_text[0]
        return self._cleanup_answer(raw_answer)


if __name__ == "__main__":
    model = QwenModel()
    img = Image.new('RGB', (100, 100), color = 'green')
    ans = model.generate_answer(img, "What color is this?")
    print("Answer:", ans)
