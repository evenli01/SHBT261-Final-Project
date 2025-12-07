#!/bin/bash

echo "Running Qwen Fine-tuning with LoRA..."
echo "====================================="

EPOCHS=3
BATCH_SIZE=4      # per-GPU batch size
GRAD_ACCUM=2      # effective batch size = 4 * 2 = 8
LR=1e-4
CUDA_ID=0         # adjust for your RunPod GPU index
# LIMIT="--limit 100"  # uncomment for quick small-scale runs
LIMIT=""

echo ""
echo "=== Fine-tuning Qwen2.5-VL-3B-Instruct with LoRA ==="
python scripts/train.py \
  --model qwen \
  --output_dir checkpoints/qwen_lora \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --cuda_id $CUDA_ID \
  $LIMIT

echo ""
echo "Qwen fine-tuning complete!"
echo "Checkpoints saved in checkpoints/qwen_lora"
