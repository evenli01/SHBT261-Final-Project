"""
Zero-shot evaluation of Qwen2.5-VL-3B on TextVQA dataset.
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


def evaluate_zero_shot(
    model_wrapper: QwenVLModel,
    dataloader: DataLoader,
    metrics_calc: TextVQAMetrics,
    device: str = "cuda",
    max_samples: int = None,
    save_predictions: bool = True,
    output_file: str = None
) -> dict:
    """
    Perform zero-shot evaluation on a dataset.
    
    Args:
        model_wrapper: QwenVLModel instance
        dataloader: DataLoader for evaluation data
        metrics_calc: TextVQAMetrics instance
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None for all)
        save_predictions: Whether to save predictions
        output_file: File to save predictions to
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    model = model_wrapper.get_model()
    processor = model_wrapper.get_processor()
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_questions = []
    all_question_ids = []
    
    print(f"\nStarting zero-shot evaluation...")
    print(f"Total batches: {len(dataloader)}")
    
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and num_samples >= max_samples:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Handle pixel values - now always a padded tensor
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # Generate predictions
            try:
                outputs = model_wrapper.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Decode predictions
                for i in range(len(outputs)):
                    # Skip input tokens, decode only generated tokens
                    generated_ids = outputs[i][len(input_ids[i]):]
                    prediction = processor.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()
                    
                    all_predictions.append(prediction)
                    all_ground_truths.append(batch["answers"][i])
                    all_questions.append(batch["questions"][i])
                    all_question_ids.append(batch["question_ids"][i])
                    
                    num_samples += 1
                    
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                # Add dummy predictions for failed batches
                for i in range(len(batch["questions"])):
                    all_predictions.append("")
                    all_ground_truths.append(batch["answers"][i])
                    all_questions.append(batch["questions"][i])
                    all_question_ids.append(batch["question_ids"][i])
                continue
    
    print(f"\nEvaluated {len(all_predictions)} samples")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = metrics_calc.compute_all_metrics(
        predictions=all_predictions,
        ground_truths_list=all_ground_truths,
        questions=all_questions if metrics_calc.use_llm_judge else None
    )
    
    # Prepare results
    results = {
        "metrics": metrics,
        "num_samples": len(all_predictions),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save predictions if requested
    if save_predictions and output_file:
        predictions_data = []
        for i in range(len(all_predictions)):
            predictions_data.append({
                "question_id": str(all_question_ids[i]),
                "question": all_questions[i],
                "prediction": all_predictions[i],
                "ground_truths": all_ground_truths[i],
                "correct": metrics_calc.compute_exact_match(
                    all_predictions[i], 
                    all_ground_truths[i]
                ) > 0.5
            })
        
        results["predictions"] = predictions_data
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on TextVQA")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/zero_shot",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM as a judge for evaluation"
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for LLM judge"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./textvqa_data",
        help="Directory containing TextVQA data"
    )
    parser.add_argument(
        "--use_hf_direct",
        action="store_true",
        default=True,
        help="Load dataset directly from HuggingFace"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Zero-Shot Evaluation - Qwen2.5-VL-3B on TextVQA")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    
    # Initialize model
    print("\n[1/4] Loading model...")
    config = ModelConfig()
    config.model_name = args.model_name
    config.use_lora = False  # No LoRA for zero-shot
    config.per_device_eval_batch_size = args.batch_size
    
    model_wrapper = QwenVLModel(config)
    
    # Load data
    print("\n[2/4] Loading dataset...")
    dataset = load_textvqa_data(
        data_dir=args.data_dir,
        split=args.split,
        use_hf_direct=args.use_hf_direct
    )
    
    # Create dataset and dataloader
    eval_dataset = TextVQADataset(
        dataset=dataset,
        processor=model_wrapper.get_processor(),
        split=args.split
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize metrics
    print("\n[3/4] Initializing metrics...")
    metrics_calc = TextVQAMetrics(
        use_llm_judge=args.use_llm_judge,
        llm_api_key=args.llm_api_key
    )
    
    # Evaluate
    print("\n[4/4] Running evaluation...")
    output_file = os.path.join(
        args.output_dir,
        f"zero_shot_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    results = evaluate_zero_shot(
        model_wrapper=model_wrapper,
        dataloader=eval_loader,
        metrics_calc=metrics_calc,
        device=args.device,
        max_samples=args.max_samples,
        save_predictions=True,
        output_file=output_file
    )
    
    # Print metrics
    print_metrics(results["metrics"], f"Zero-Shot Evaluation - {args.split.capitalize()} Set")
    
    print("\n" + "=" * 70)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
