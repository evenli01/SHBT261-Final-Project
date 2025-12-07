#!/bin/bash

echo "Running Fine-tuned Qwen Evaluation..."
echo "====================================="

LORA_PATH="checkpoints/qwen_lora"
# LIMIT="--limit 100"  # optional
LIMIT=""

echo ""
echo "=== Qwen LoRA: Baseline (no OCR prompt) ==="
python scripts/run_eval.py --model qwen --lora_path $LORA_PATH $LIMIT

echo ""
echo "=== Qwen LoRA: Descriptive prompt (no OCR) ==="
python scripts/run_eval.py --model qwen --lora_path $LORA_PATH --prompt_template descriptive $LIMIT

echo ""
echo "=== Qwen LoRA: Text-focus prompt (no OCR) ==="
python scripts/run_eval.py --model qwen --lora_path $LORA_PATH --prompt_template text_focus $LIMIT

echo ""
echo "=== Qwen LoRA: Basic OCR prompt ==="
python scripts/run_eval.py --model qwen --lora_path $LORA_PATH --prompt_template basic_ocr $LIMIT

echo ""
echo "=== Qwen LoRA: Structured OCR prompt ==="
python scripts/run_eval.py --model qwen --lora_path $LORA_PATH --prompt_template structured_ocr $LIMIT

echo ""
echo "Fine-tuned Qwen evaluation complete!"
echo "Results saved in results/ directory"
