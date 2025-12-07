# scripts/plot_results.py

import os
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def parse_filename(fname):
    """
    Expected patterns like:
      qwen_validation_results.json
      qwen_prompt_descriptive_validation_results.json
      qwen_finetuned_validation_results.json
      qwen_finetuned_prompt_basic_ocr_validation_results.json
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")

    model = parts[0]  # qwen
    finetuned = "finetuned" in parts
    prompt = None
    split = "validation"

    if "prompt" in parts:
        idx = parts.index("prompt")
        if idx + 1 < len(parts):
            prompt = parts[idx + 1]

    # last part before "results" is split
    if parts[-1] == "results" and len(parts) >= 3:
        split = parts[-2]

    return {
        "model": model,
        "finetuned": finetuned,
        "prompt": prompt or "none",
        "split": split,
    }


def load_results():
    records = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("qwen"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)
        meta = parse_filename(fname)
        metrics = data.get("metrics", {})
        record = {
            "file": fname,
            "finetuned": meta["finetuned"],
            "prompt": meta["prompt"],
            "split": meta["split"],
            "accuracy": metrics.get("accuracy", 0.0),
            "bleu": metrics.get("bleu", 0.0),
            "meteor": metrics.get("meteor", 0.0),
            "rouge1": metrics.get("rouge1", 0.0),
            "semantic_similarity": metrics.get("semantic_similarity", 0.0),
        }
        records.append(record)
    return records


def plot_bar(records, metric, title, filename):
    labels = []
    values = []

    for r in records:
        label = f"{'FT' if r['finetuned'] else 'ZS'}-{r['prompt']}"
        labels.append(label)
        values.append(r.get(metric, 0.0))

    if not labels:
        print(f"No records to plot for metric {metric}")
        return

    plt.figure(figsize=(max(8, len(labels) * 0.6), 5))
    x = range(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    records = load_results()
    if not records:
        print("No qwen*.json result files found in results/. Nothing to plot.")
        return

    # Sort by finetuned -> prompt for nicer ordering
    records.sort(key=lambda r: (r["finetuned"], r["prompt"]))

    plot_bar(records, "accuracy", "Qwen TextVQA Accuracy (ZS vs FT, prompts)", "qwen_accuracy.png")
    plot_bar(records, "bleu", "Qwen BLEU (ZS vs FT, prompts)", "qwen_bleu.png")
    plot_bar(records, "meteor", "Qwen METEOR (ZS vs FT, prompts)", "qwen_meteor.png")
    plot_bar(records, "rouge1", "Qwen ROUGE-1 (ZS vs FT, prompts)", "qwen_rouge1.png")
    plot_bar(
        records,
        "semantic_similarity",
        "Qwen Semantic Similarity (ZS vs FT, prompts)",
        "qwen_semantic_similarity.png",
    )


if __name__ == "__main__":
    main()
