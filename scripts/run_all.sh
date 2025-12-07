#!/bin/bash

echo "Starting Complete Qwen Experiment Pipeline..."
echo "============================================="

echo ""
echo "Step 1: Zero-shot Qwen Evaluation"
bash scripts/run_zero_shot_qwen.sh

echo ""
echo "Step 2: Fine-tuning Qwen with LoRA"
bash scripts/run_fine_tuning_qwen.sh

echo ""
echo "Step 3: Fine-tuned Qwen Evaluation"
bash scripts/run_finetuned_eval_qwen.sh

echo ""
echo "Step 4: Generating Result Visualizations"
python scripts/plot_results.py

echo ""
echo "========================================="
echo "All Qwen experiments complete!"
echo "Check results/ and results/plots/ for outputs."
