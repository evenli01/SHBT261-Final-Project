#!/bin/bash

echo "Running Comprehensive Zero-shot Evaluation for Qwen..."
echo "======================================================="

# Optional: limit for quick debugging
# LIMIT="--limit 100"
LIMIT=""

echo ""
echo "=== Qwen Zero-shot: No OCR, baseline prompt ==="
python scripts/run_eval.py --model qwen $LIMIT

echo ""
echo "=== Qwen Zero-shot: Descriptive prompt (no OCR) ==="
python scripts/run_eval.py --model qwen --prompt_template descriptive $LIMIT

echo ""
echo "=== Qwen Zero-shot: Text-focus prompt (no OCR) ==="
python scripts/run_eval.py --model qwen --prompt_template text_focus $LIMIT

echo ""
echo "=== Qwen Zero-shot: Basic OCR prompt ==="
# Basic OCR = cleans OCR tokens and injects them as flat text
python scripts/run_eval.py --model qwen --prompt_template basic_ocr $LIMIT

echo ""
echo "=== Qwen Zero-shot: Structured OCR prompt ==="
# Structured OCR = category-aware OCR summarization
python scripts/run_eval.py --model qwen --prompt_template structured_ocr $LIMIT

echo ""
echo "Zero-shot evaluation complete!"
echo "Results saved in results/ directory"
