"""
Zero-shot evaluation of Qwen2.5-VL-3B on TextVQA (local dataset).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.qwen_model import QwenVLModel
from models.model_config import ModelConfig
from data.preprocess import load_textvqa_data, TextVQADataset, collate_fn
from evaluation.metrics import TextVQAMetrics, print_metrics


# ============================================================
#                   Evaluation Step
# ============================================================

def evaluate_zero_shot(
    model_wrapper: QwenVLModel,
    dataloader,
    metrics_calc,
    device="cuda",
    max_samples=None,
    save_predictions=True,
    output_file=None
):
    model = model_wrapper.get_model()
    processor = model_wrapper.get_processor()
    model.eval()

    preds, gts, questions, qids = [], [], [], []
    n = 0

    print("\nStarting zero-shot evaluation...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if max_samples and n >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # Generate predictions
            outputs = model_wrapper.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for i in range(len(outputs)):
                gen_ids = outputs[i][len(input_ids[i]):]
                pred = processor.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                ).strip()

                preds.append(pred)
                gts.append(batch["answers"][i])
                questions.append(batch["questions"][i])
                qids.append(batch["question_ids"][i])

                n += 1
                if max_samples and n >= max_samples:
                    break

    # Compute metrics
    print("\nComputing metrics...")
    metrics = metrics_calc.compute_all_metrics(
        predictions=preds,
        ground_truths_list=gts,
        questions=questions if metrics_calc.use_llm_judge else None
    )

    results = {
        "metrics": metrics,
        "num_samples": len(preds),
        "timestamp": datetime.now().isoformat()
    }

    # Save predictions
    if save_predictions and output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out = []
        for i in range(len(preds)):
            out.append({
                "question_id": str(qids[i]),
                "question": questions[i],
                "prediction": preds[i],
                "ground_truths": gts[i],
                "correct": metrics_calc.compute_exact_match(preds[i], gts[i]) > 0.5
            })

        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved predictions to {output_path}")

    return results


# ============================================================
#                   Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="./results/zero_shot")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data_dir", type=str,
                        default="./textvqa_data")

    # FIX: allow HF direct load only when explicitly requested
    parser.add_argument("--use_hf_direct",
                        action="store_true",
                        help="Explicitly load from HF (not recommended)")
    parser.set_defaults(use_hf_direct=False)

    parser.add_argument("--use_llm_judge", action="store_true")
    parser.add_argument("--llm_api_key", type=str, default=None)

    args = parser.parse_args()

    print("="*70)
    print(f"Zero-Shot Evaluation — {args.model_name}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print("="*70)

    # Load model
    config = ModelConfig()
    config.model_name = args.model_name
    config.use_lora = False

    model_wrapper = QwenVLModel(config)

    # Load dataset **locally**
    dataset = load_textvqa_data(args.data_dir, args.split,
                                use_hf_direct=args.use_hf_direct)

    eval_dataset = TextVQADataset(dataset, model_wrapper.get_processor(),
                                  split=args.split)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Metrics
    metrics_calc = TextVQAMetrics(
        use_llm_judge=args.use_llm_judge,
        llm_api_key=args.llm_api_key
    )

    # Eval
    out_file = (
        f"{args.output_dir}/zero_shot_{args.split}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    results = evaluate_zero_shot(
        model_wrapper,
        eval_loader,
        metrics_calc,
        device=args.device,
        max_samples=args.max_samples,
        save_predictions=True,
        output_file=out_file
    )

    print_metrics(results["metrics"], f"Zero-shot {args.split}")

    print("="*70)
    print("Done.")
    print("="*70)


if __name__ == "__main__":
    main()
