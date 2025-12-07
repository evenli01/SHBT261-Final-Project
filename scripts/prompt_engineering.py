# scripts/prompt_engineering.py

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen import QwenModel
from utils.dataset import TextVQADataset


def test_prompts():
    prompts = [
        "Question: {question} Answer:",
        "Based on the image, answer briefly: {question}",
        "Look at the image and answer in a few words: {question}",
        "{question}",
        "Answer in 1–3 words: {question}",
        "Focus on any visible text in the image. Question: {question}",
    ]

    dataset = TextVQADataset(split="validation")
    samples = [dataset[i] for i in range(5)]

    model = QwenModel()
    print("Testing prompt variants on Qwen...")

    for p_idx, template in enumerate(prompts):
        print(f"\n--- Prompt Template {p_idx+1}: '{template}' ---")
        for s_idx, sample in enumerate(samples):
            question = sample["question"]
            formatted_question = template.format(question=question)
            try:
                ans = model.generate_answer(sample["image"], formatted_question)
                print(f"Sample {s_idx} Q: {question} | A: {ans}")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    test_prompts()
