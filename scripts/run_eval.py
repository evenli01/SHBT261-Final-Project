# scripts/run_eval.py

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from utils.dataset import TextVQADataset
from utils.metrics import calculate_metrics
from models.qwen import QwenModel
import json
from peft import PeftModel
import re


# ---------- OCR utilities (same as classmate) ----------

def clean_ocr_text(ocr_tokens, max_tokens=15, min_token_length=2):
    if not ocr_tokens:
        return ""
    cleaned = []
    seen = set()
    for token in ocr_tokens:
        token = token.strip()
        if not token:
            continue
        if len(token) < min_token_length:
            continue
        alnum_ratio = sum(c.isalnum() for c in token) / len(token) if token else 0
        if alnum_ratio < 0.3:
            continue
        token_lower = token.lower()
        if token_lower not in seen:
            seen.add(token_lower)
            cleaned.append(token)
        if len(cleaned) >= max_tokens:
            break
    return " ".join(cleaned)


def classify_question(question):
    q = question.lower()
    brand_keywords = ["brand", "logo", "label", "company", "manufacturer", "maker"]
    if any(k in q for k in brand_keywords):
        return "brand"

    number_keywords = ["number", "percent", "%", "price", "$", "cost", "amount", "how much", "how many"]
    if any(k in q for k in number_keywords):
        return "number"

    date_keywords = ["year", "date", "born", "since", "when"]
    if any(k in q for k in date_keywords):
        return "date"

    time_keywords = ["time", "clock", "what time"]
    if any(k in q for k in time_keywords):
        return "time"

    text_keywords = ["text", "say", "spell", "read", "what does", "what is written"]
    if any(k in q for k in text_keywords):
        return "text"

    return "general"


def summarize_ocr_by_type(ocr_tokens, q_type, max_tokens=15, min_token_length=2):
    if not ocr_tokens:
        return ""
    filtered_tokens = []
    for token in ocr_tokens:
        token = token.strip()
        if not token or len(token) < min_token_length:
            continue
        alnum_ratio = sum(c.isalnum() for c in token) / len(token) if token else 0
        if alnum_ratio < 0.3:
            continue

        if q_type == "brand":
            if token[0].isupper() and token.isalpha():
                filtered_tokens.append(token)
        elif q_type == "number":
            if any(c.isdigit() for c in token) or "$" in token or "%" in token:
                filtered_tokens.append(token)
        elif q_type == "date":
            if re.match(r"^\d{4}$", token) or re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", token):
                filtered_tokens.append(token)
            elif token.isdigit() and len(token) >= 4:
                filtered_tokens.append(token)
        elif q_type == "time":
            if re.match(r"^\d{1,2}[:]\d{2}", token):
                filtered_tokens.append(token)
            elif any(c.isdigit() for c in token) and len(token) <= 5:
                filtered_tokens.append(token)
        elif q_type == "text":
            filtered_tokens.append(token)
        else:
            filtered_tokens.append(token)

        if len(filtered_tokens) >= max_tokens:
            break

    return " ".join(filtered_tokens) if filtered_tokens else ""


# ---------- Prompt templates ----------

PROMPT_TEMPLATES = {
    # Baselines (non-OCR)
    "default": "Question: {question} Answer:",
    "descriptive": "Based on the image, answer the question briefly: {question}",
    "instruction": "Look at the image and answer in a few words: {question}",
    "direct": "{question}",
    "text_focus": "Focus on any visible text in the image. Question: {question}",

    # Slightly tweaked short-answer style (not in your classmate’s file)
    "short_direct": "Answer in 1–3 words: {question}",

    # OCR-related templates (from classmate + aliases for your ablation)
    "ocr_hint": "The image contains the following text: {ocr_text}. Question: {question} Answer:",
    "ocr_hint_v3": "Answer this question about the image: {question}\nVisible text in image: {ocr_text}\nAnswer:",

    # New friendly aliases for your ablation:
    # Basic OCR → use cleaned OCR tokens directly
    "basic_ocr": "Detected text in the image: {ocr_text}\nQuestion: {question}\nAnswer in a short phrase:",

    # Structured OCR → category-aware summary
    "ocr_category": (
        "Relevant text in the image ({q_type}): {ocr_summary}\n"
        "Question: {question}\n"
        "Use ONLY that text when answering.\n"
        "Answer briefly:"
    ),
    "structured_ocr": (
        "Relevant text in the image ({q_type}): {ocr_summary}\n"
        "Question: {question}\n"
        "Answer in 1–3 words using that text:"
    ),
}


def get_qwen_model(lora_path=None):
    model = QwenModel()
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model.model = PeftModel.from_pretrained(model.model, lora_path)
        print("LoRA adapter loaded successfully!")
    return model


def evaluate(args):
    dataset = TextVQADataset(split=args.split)

    # Get prompt template string
    if args.prompt_template:
        if args.prompt_template in PROMPT_TEMPLATES:
            prompt_template = PROMPT_TEMPLATES[args.prompt_template]
            print(f"Using prompt template '{args.prompt_template}': {prompt_template}")
        else:
            prompt_template = args.prompt_template
            print(f"Using custom prompt template: {prompt_template}")
    else:
        prompt_template = None
        print("Using raw question (no explicit template).")

    model = get_qwen_model(args.lora_path)
    results = []

    print(f"Starting evaluation on {args.split} split with {len(dataset)} samples...")

    indices = range(args.limit) if args.limit else range(len(dataset))

    for i in tqdm(indices):
        sample = dataset[i]
        image = sample["image"]
        question = sample["question"]
        ground_truth_answers = sample["answers"]
        image_id = sample["image_id"]
        ocr_tokens = sample.get("ocr_tokens", [])

        ocr_text = ""
        ocr_summary = ""
        q_type = ""

        # Decide OCR usage based on template
        if prompt_template:
            if "{ocr_summary}" in prompt_template and "{q_type}" in prompt_template:
                q_type = classify_question(question)
                ocr_summary = summarize_ocr_by_type(ocr_tokens, q_type)
            elif "{ocr_text}" in prompt_template:
                ocr_text = clean_ocr_text(ocr_tokens)

        # Build formatted question
        if prompt_template:
            if "{ocr_summary}" in prompt_template and "{q_type}" in prompt_template:
                if ocr_summary:
                    formatted_question = prompt_template.format(
                        question=question, ocr_summary=ocr_summary, q_type=q_type
                    )
                else:
                    # Fallback: strip OCR header lines if no OCR available
                    stripped = (
                        prompt_template.replace(
                            "Relevant text in the image ({q_type}): {ocr_summary}\n", ""
                        )
                        .replace("Use ONLY that text when answering.\n", "")
                        .replace("Answer in 1–3 words using that text:", "Answer briefly:")
                    )
                    formatted_question = stripped.format(question=question, q_type=q_type)
            elif "{ocr_text}" in prompt_template:
                if ocr_text:
                    formatted_question = prompt_template.format(
                        question=question, ocr_text=ocr_text
                    )
                else:
                    stripped = (
                        prompt_template.replace("{ocr_text}", "")
                        .replace("Detected text in the image:", "")
                        .replace("Visible text in image:", "")
                    )
                    formatted_question = stripped.format(question=question)
            else:
                formatted_question = prompt_template.format(question=question)
        else:
            formatted_question = question

        try:
            predicted_answer = model.generate_answer(image, formatted_question)
        except (RuntimeError, torch.cuda.CudaError) as e:
            err_str = str(e)
            print(f"\nRuntime/CUDA error for image {image_id}: {err_str}")
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            predicted_answer = ""
        except Exception as e:
            print(f"Error generating answer for image {image_id}: {e}")
            predicted_answer = ""

        result_item = {
            "image_id": image_id,
            "question": question,
            "formatted_question": formatted_question,
            "predicted_answer": predicted_answer,
            "ground_truth_answers": ground_truth_answers,
        }
        if ocr_summary:
            result_item["ocr_summary"] = ocr_summary
            result_item["q_type"] = q_type
        if ocr_text:
            result_item["ocr_text"] = ocr_text

        results.append(result_item)

        if (i + 1) % 50 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    metrics = calculate_metrics(results)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        # Skip per-category dict here; we'll print it separately
        if k == "per_category":
            continue
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Pretty-print per-category breakdown
    per_cat = metrics.get("per_category", {})
    if per_cat:
        print("\nPer-Category Metrics:")
        for cat, stats in per_cat.items():
            print(f"  [{cat}]")
            for sk, sv in stats.items():
                if isinstance(sv, (int, float)):
                    print(f"    {sk}: {sv:.4f}")
                else:
                    print(f"    {sk}: {sv}")

    os.makedirs("results", exist_ok=True)
    filename_parts = ["qwen"]
    if args.lora_path:
        filename_parts.append("finetuned")
    if args.prompt_template:
        template_name = (
            args.prompt_template
            if args.prompt_template in PROMPT_TEMPLATES
            else "custom"
        )
        filename_parts.append(f"prompt_{template_name}")
    filename_parts.append(args.split)
    filename_parts.append("results")

    output_file = f"results/{'_'.join(filename_parts)}.json"
    output_data = {
        "metrics": metrics,
        "results": results,
        "config": {
            "model": "qwen",
            "lora_path": args.lora_path,
            "prompt_template": prompt_template,
            "prompt_template_name": args.prompt_template,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL evaluation on TextVQA with LoRA and prompt/OCR ablations"
    )
    parser.add_argument("--model", type=str, required=True, choices=["qwen"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help=(
            "Prompt template to use. Options: "
            + ", ".join(PROMPT_TEMPLATES.keys())
            + ", or a custom format string with {question}, {ocr_text}, {ocr_summary}, {q_type}."
        ),
    )
    args = parser.parse_args()
    evaluate(args)
