# scripts/train.py

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(args):
    print(f"Starting fine-tuning for {args.model}...")

    if args.model != "qwen":
        raise ValueError("This train.py is Qwen-only. Use --model qwen.")

    # 4-bit quantization config (same as your classmate) :contentReference[oaicite:4]{index=4}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": f"cuda:{args.cuda_id}"},
        trust_remote_code=True,
    )

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    task_type = TaskType.CAUSAL_LM

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = load_dataset("lmms-lab/textvqa", split="train")
    if args.limit:
        dataset = dataset.select(range(args.limit))

    from qwen_vl_utils import process_vision_info  # required by processor

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answers"][0] for item in batch]

        all_messages = []
        for q, a, img in zip(questions, answers, images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Question: {q} Answer: {a}"},
                    ],
                }
            ]
            all_messages.append(messages)

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in all_messages
        ]

        image_inputs_list = [[msg[0]["content"][0]["image"]] for msg in all_messages]

        inputs = processor(
            text=texts,
            images=image_inputs_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        for i, (q, a) in enumerate(zip(questions, answers)):
            answer_tokens = processor.tokenizer(a, add_special_tokens=False)["input_ids"]
            seq_len = (labels[i] != processor.tokenizer.pad_token_id).sum()
            answer_start = max(0, seq_len - len(answer_tokens) - 5)
            labels[i, :answer_start] = -100

        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    gradient_accumulation_steps = args.grad_accum_steps
    model.train()
    total_steps = len(dataloader) * args.epochs
    progress_bar = tqdm(total=total_steps, desc="Training")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            try:
                outputs = model(**batch)
                loss = outputs.loss

                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss at step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue

                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "epoch": f"{epoch+1}/{args.epochs}",
                    }
                )

            except torch.cuda.OutOfMemoryError as e:
                print(f"\nCUDA OOM at step {step}: {e}")
                print("Clearing cache and skipping batch...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"\nError in batch: {e}")
                optimizer.zero_grad()
                continue

        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")

        epoch_output_dir = f"{args.output_dir}/epoch_{epoch+1}"
        model.save_pretrained(epoch_output_dir)
        print(f"Checkpoint saved to {epoch_output_dir}")

    progress_bar.close()
    model.save_pretrained(args.output_dir)
    print(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", help="Model to fine-tune (qwen only)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda_id", type=int, default=0)
    args = parser.parse_args()
    train(args)
