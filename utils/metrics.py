import re
import collections

import evaluate
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# -------------------------------------------------------------------
# Global metric + model loading
# -------------------------------------------------------------------

try:
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
except Exception as e:
    print(f"Warning: Failed to load text metrics: {e}")
    bleu_metric = None
    meteor_metric = None
    rouge_metric = None

try:
    # Match your classmate: use CPU (you can change to cuda later if you want)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    print(f"Warning: Failed to load semantic model: {e}")
    semantic_model = None


# -------------------------------------------------------------------
# VQA-style preprocessing + accuracy
# -------------------------------------------------------------------

def preprocess_answer(answer):
    """
    Standard VQA text preprocessing.
    """
    answer = str(answer).lower()
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    return answer


def compute_vqa_accuracy(ground_truth_list, predicted_answer):
    """
    Compute VQA-style consensus accuracy.
    Official metric: acc = min(1, (number of matching human answers) / 3).
    Matching is exact string match after normalization.
    """
    if not ground_truth_list:
        return 0.0

    predicted_answer = preprocess_answer(predicted_answer)
    ground_truth_list = [preprocess_answer(ans) for ans in ground_truth_list]

    match_count = sum(1 for gt in ground_truth_list if gt == predicted_answer)
    return min(1.0, match_count / 3.0)


# -------------------------------------------------------------------
# Semantic similarity (per-example, same style as classmate)
# -------------------------------------------------------------------

def compute_semantic_similarity(ground_truth_list, predicted_answer):
    """
    Compute semantic similarity between predicted answer and ground truth answers.
    Returns the maximum similarity score among all ground truth answers.
    """
    if semantic_model is None:
        return 0.0

    predicted_answer = str(predicted_answer)
    ground_truth_list = [str(gt) for gt in ground_truth_list] if ground_truth_list else [""]

    # Encode
    pred_emb = semantic_model.encode(predicted_answer, convert_to_tensor=True)
    gt_embs = semantic_model.encode(ground_truth_list, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(pred_emb, gt_embs)

    if cosine_scores.numel() == 0:
        return 0.0

    return float(torch.max(cosine_scores).item())


# -------------------------------------------------------------------
# Aggregate metrics (same set as your classmate)
# -------------------------------------------------------------------

def calculate_metrics(results):
    """
    Calculate average metrics over the results.

    Args:
        results: list of dicts with keys:
            - 'predicted_answer'
            - 'ground_truth_answers'

    Returns:
        dict of metrics:
            - accuracy
            - bleu
            - meteor
            - rouge1, rouge2, rougeL
            - semantic_similarity
    """
    if not results:
        return {}

    total_acc = 0.0
    predictions = []
    references = []
    semantic_scores = []

    for item in results:
        pred = item["predicted_answer"]
        gts = item["ground_truth_answers"]

        # VQA accuracy
        acc = compute_vqa_accuracy(gts, pred)
        total_acc += acc

        predictions.append(str(pred))
        references.append([str(gt) for gt in gts] if gts else [""])

        # Semantic similarity (max over GTs)
        if semantic_model is not None:
            sim = compute_semantic_similarity(gts, pred)
            semantic_scores.append(sim)

    metrics = {
        "accuracy": total_acc / len(results)
    }

    # BLEU
    if bleu_metric is not None:
        try:
            bleu_score = bleu_metric.compute(
                predictions=predictions,
                references=references,
                max_order=2,  # bigrams: more reasonable for short VQA answers
                smooth=True,
            )
            metrics["bleu"] = bleu_score.get("bleu", 0.0)
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            metrics["bleu"] = 0.0

    # METEOR
    if meteor_metric is not None:
        try:
            meteor_score = meteor_metric.compute(
                predictions=predictions,
                references=references,
            )
            metrics["meteor"] = meteor_score.get("meteor", 0.0)
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            metrics["meteor"] = 0.0

    # ROUGE
    if rouge_metric is not None:
        try:
            rouge_score = rouge_metric.compute(
                predictions=predictions,
                references=references,
            )
            metrics["rouge1"] = rouge_score.get("rouge1", 0.0)
            metrics["rouge2"] = rouge_score.get("rouge2", 0.0)
            metrics["rougeL"] = rouge_score.get("rougeL", 0.0)
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            metrics["rouge1"] = 0.0
            metrics["rouge2"] = 0.0
            metrics["rougeL"] = 0.0

    # Semantic similarity (mean over examples)
    if semantic_scores:
        metrics["semantic_similarity"] = float(sum(semantic_scores) / len(semantic_scores))
    else:
        metrics["semantic_similarity"] = 0.0

    return metrics

