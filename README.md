# SHBT261-Final-Project

Author: Yiwen Li, Elena Xing, Nouraldeen Ibrahim, Xu Han

A comprehensive study on Vision-Language Models (VLMs) for Text-based Visual Question Answering (TextVQA), featuring BLIP2 and Qwen2.5-VL with LoRA fine-tuning, prompt engineering, and OCR ablation experiments.

## 📋 Project Overview

This project explores the effectiveness of vision-language models on text-heavy image understanding tasks using the TextVQA dataset. The research focuses on:

- **Model Comparison**: BLIP2 (Salesforce/blip2-opt-2.7b) vs. Qwen2.5-VL-3B-Instruct
- **Fine-tuning Strategy**: LoRA (Low-Rank Adaptation) with 4-bit quantization for efficient training
- **Prompt Engineering**: Multiple prompt templates to optimize model performance
- **OCR Integration**: Basic and structured OCR token utilization for improved text recognition
- **Comprehensive Evaluation**: BLEU, ROUGE, METEOR, Exact Match, and Semantic Similarity metrics

## 🚀 Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/evenli01/SHBT261-Final-Project.git
cd SHBT261-Final-Project
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data** (for metrics)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Hardware Requirements

- **Minimum**: 16GB GPU VRAM for inference
- **Recommended**: 24GB+ GPU VRAM for fine-tuning
- **Note**: The code uses 4-bit quantization to reduce memory requirements

## 📖 Usage

### Complete Pipeline

Run all experiments (zero-shot, fine-tuning, evaluation, and visualization):

```bash
bash scripts/run_all.sh
```

This executes the following steps:
1. Zero-shot evaluation with multiple prompts
2. LoRA fine-tuning on training set
3. Fine-tuned model evaluation
4. Result visualization and plotting

### Individual Components

#### 1. Zero-Shot Evaluation

Evaluate Qwen2.5-VL with various prompt templates:

```bash
bash scripts/run_zero_shot_qwen.sh
```

Or run individual prompt templates:

```bash
# No prompt template (baseline)
python scripts/run_eval.py --model qwen

# Descriptive prompt
python scripts/run_eval.py --model qwen --prompt_template descriptive

# Basic OCR-aware prompt
python scripts/run_eval.py --model qwen --prompt_template basic_ocr

# Structured OCR prompt (category-aware)
python scripts/run_eval.py --model qwen --prompt_template structured_ocr

# Text-focus prompt
python scripts/run_eval.py --model qwen --prompt_template text_focus
```

Available prompt templates:
- `default`: Basic question-answer format
- `descriptive`: Encourages detailed image description
- `text_focus`: Emphasizes visible text in images
- `basic_ocr`: Includes cleaned OCR tokens
- `structured_ocr`: Category-aware OCR summarization (brand, number, date, time, text, general)

#### 2. Fine-Tuning

Fine-tune Qwen2.5-VL with LoRA on TextVQA training set:

```bash
bash scripts/run_fine_tuning_qwen.sh
```

Or with custom parameters:

```bash
python scripts/train.py \
  --model qwen \
  --output_dir checkpoints/qwen_lora \
  --epochs 3 \
  --lr 1e-4 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --cuda_id 0
```

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 1)
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size per GPU (default: 4)
- `--grad_accum_steps`: Gradient accumulation steps (default: 2)
- `--limit`: Limit training samples for quick testing
- `--cuda_id`: CUDA device ID (default: 0)

#### 3. Fine-Tuned Model Evaluation

Evaluate the fine-tuned model:

```bash
bash scripts/run_finetuned_eval_qwen.sh
```

Or evaluate with specific LoRA checkpoint:

```bash
python scripts/run_eval.py \
  --model qwen \
  --lora_path checkpoints/qwen_lora \
  --prompt_template structured_ocr
```

#### 4. Generate Visualizations

Create comparison plots from evaluation results:

```bash
python scripts/plot_results.py
```

Generates plots in `results/plots/`:
- BLEU score comparison
- ROUGE-1 F1 score comparison
- METEOR score comparison
- Exact Match accuracy comparison
- Semantic similarity comparison

### BLIP2 Experiments

BLIP2 experiments are available in the Jupyter notebook:

```bash
jupyter notebook BLIP2/BLIP2.ipynb
```

The notebook includes:
- BLIP2 model setup and fine-tuning
- Zero-shot and fine-tuned evaluations
- Prompt engineering experiments
- OCR ablation studies

## 📂 File Structure

```
SHBT261-Final-Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── BLIP2/                            # BLIP2 experiments
│   ├── BLIP2.ipynb                   # Main Jupyter notebook
│   └── BLIP2 Results/                # BLIP2 evaluation results
│       ├── adapter_config.json       # LoRA adapter configuration
│       ├── adapter_model.safetensors # Fine-tuned LoRA weights
│       ├── *.json                    # Evaluation result JSONs
│       ├── *.png                     # Performance visualization plots
│       └── README.md                 # Model card
│
├── models/                           # Model implementations
│   ├── base_model.py                 # Base VLM interface
│   └── qwen.py                       # Qwen2.5-VL wrapper
│
├── utils/                            # Utility modules
│   ├── dataset.py                    # TextVQA dataset loader
│   └── metrics.py                    # Evaluation metrics (BLEU, ROUGE, etc.)
│
├── scripts/                          # Experiment scripts
│   ├── run_all.sh                    # Complete pipeline runner
│   ├── run_zero_shot_qwen.sh         # Zero-shot evaluation
│   ├── run_fine_tuning_qwen.sh       # LoRA fine-tuning
│   ├── run_finetuned_eval_qwen.sh    # Fine-tuned evaluation
│   ├── train.py                      # Training script
│   ├── run_eval.py                   # Evaluation script
│   ├── prompt_engineering.py         # Prompt template utilities
│   └── plot_results.py               # Visualization generator
│
├── checkpoints/                      # Model checkpoints
│   └── qwen_lora/                    # Qwen LoRA adapters
│       ├── adapter_config.json       # LoRA configuration
│       ├── adapter_model.safetensors # Final LoRA weights
│       ├── epoch_1/                  # Epoch 1 checkpoint
│       ├── epoch_2/                  # Epoch 2 checkpoint
│       └── epoch_3/                  # Epoch 3 checkpoint
│
└── results/                          # Evaluation results
    ├── qwen_*.json                   # Evaluation result files
    └── plots/                        # Performance comparison plots
        ├── qwen_accuracy.png
        ├── qwen_bleu.png
        ├── qwen_meteor.png
        ├── qwen_rouge1.png
        └── qwen_semantic_similarity.png
```

## 📊 Evaluation Metrics

The project uses comprehensive metrics for evaluation:

1. **BLEU (Bilingual Evaluation Understudy)**: Measures n-gram overlap between prediction and reference
2. **ROUGE-1**: Unigram-based recall metric for text similarity
3. **METEOR**: Considers synonyms and stemming for better semantic matching
4. **Exact Match**: Percentage of exact string matches with ground truth
5. **Semantic Similarity**: Cosine similarity using sentence embeddings (sentence-transformers)

## 🔬 Key Features

### LoRA Fine-Tuning
- **Rank**: 8
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization**: 4-bit NF4 with FP16 compute dtype

### OCR Processing
- **Basic OCR**: Cleaned OCR tokens with noise filtering
- **Structured OCR**: Category-aware summarization (brand, number, date, time, text)
- **Token Filtering**: Minimum length and alphanumeric ratio thresholds
- **Deduplication**: Case-insensitive duplicate removal

### Prompt Templates
Multiple prompt engineering strategies:
- Baseline prompts without OCR
- OCR-enhanced prompts with cleaned tokens
- Category-specific OCR utilization
- Text-focused instructions

## 🛠️ Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--batch_size` (try 2 or 1)
- Increase `--grad_accum_steps` to maintain effective batch size
- Use `--limit` to test with fewer samples
- Ensure no other GPU-intensive processes are running

### CUDA Errors
- Check CUDA and PyTorch compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
- Update GPU drivers if necessary
- Set specific GPU: `export CUDA_VISIBLE_DEVICES=0`

### Slow Evaluation
- Use `--limit 100` for quick testing
- Ensure GPU is being utilized (check with `nvidia-smi`)
- Consider using a smaller model variant if available

## 📝 Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@inproceedings{textvqa,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and others},
  booktitle={CVPR},
  year={2019}
}

@article{blip2,
  title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author={Li, Junnan and others},
  journal={ICML},
  year={2023}
}
```

## 📄 License

This project is for educational and research purposes as part of SHBT261 coursework.

## 🤝 Contributing

This is a course project repository. For questions or issues, please contact the repository owner.

## 🙏 Acknowledgments

- **TextVQA Dataset**: lmms-lab/textvqa on Hugging Face
- **Qwen2.5-VL**: Alibaba Cloud Qwen Team
- **BLIP2**: Salesforce Research
- **PEFT Library**: Hugging Face for efficient fine-tuning
- **Transformers Library**: Hugging Face for model implementations
