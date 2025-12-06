"""
Data preprocessing utilities for TextVQA + Qwen2.5-VL.

Design:
- Dataset only returns raw image, question, answers, IDs.
- collate_fn uses Qwen's AutoProcessor with chat template to build
  model-ready tensors: input_ids, attention_mask, pixel_values, image_grid_thw.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# ============================================================
#                   Load Local TextVQA Dataset
# ============================================================

def load_textvqa_data(
    data_dir: str,
    split: str = "train",
    use_hf_direct: bool = False,
):
    """
    Load TextVQA dataset from local disk (HuggingFace arrow).

    Expected layout:
        data_dir/
            train/data/...
            validation/data/...
            test/data/...
    """
    if use_hf_direct:
        raise ValueError(
            "Direct HF loading is disabled. Use local dataset at data_dir instead."
        )

    split_path = Path(data_dir) / split / "data"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {split_path}.\n"
            f"Expected structure: {data_dir}/{split}/data/"
        )

    print(f"Loading local dataset: {split_path}")
    dataset = load_from_disk(str(split_path))
    print(f"Loaded {len(dataset)} samples from split '{split}'")
    return dataset


# ============================================================
#                       Dataset Class
# ============================================================

class TextVQADataset(Dataset):
    """
    TextVQA dataset for Qwen2.5-VL.

    This dataset:
    - DOES NOT tokenize
    - Returns only raw PIL image, question, answers, IDs
    """

    def __init__(self, dataset, split: str = "validation"):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        image = ex["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        question = ex["question"]
        answers = ex.get("answers", [])

        return {
            "image": image,
            "question": question,
            "answers": answers,
            "question_id": ex.get("question_id", idx),
            "image_id": ex.get("image_id", ""),
        }


# ============================================================
#                       Collate Function
# ============================================================

def collate_fn(
    batch: List[Dict],
    processor,
):
    """
    Collate function that:
    - Builds Qwen-style chat prompts with images
    - Uses AutoProcessor to build batched tensors
    - Does NOT manually pad or reshape vision tensors
    """

    images = []
    texts = []
    questions = []
    answers = []
    question_ids = []
    image_ids = []

    for ex in batch:
        img = ex["image"]
        q = ex["question"]
        ans = ex["answers"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images.append(img)
        texts.append(text)
        questions.append(q)
        answers.append(ans)
        question_ids.append(ex["question_id"])
        image_ids.append(ex["image_id"])

    # Let the official processor handle everything
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # These keys are exactly what Qwen2_5_VLForConditionalGeneration expects
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "questions": questions,
        "answers": answers,
        "question_ids": question_ids,
        "image_ids": image_ids,
    }


# ============================================================
#          Optional helper for train/val/test DataLoaders
# ============================================================

def create_dataloaders(
    processor,
    data_dir: str,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 4,
    subset_size: Optional[int] = None,
):
    """
    Helper if you later train the model.
    For now, you only really need the eval path.
    """
    from functools import partial

    train_ds = load_textvqa_data(data_dir, "train")
    val_ds = load_textvqa_data(data_dir, "validation")
    test_ds = load_textvqa_data(data_dir, "test")

    if subset_size and subset_size < len(train_ds):
        idx = np.random.choice(len(train_ds), subset_size, replace=False)
        train_ds = train_ds.select(idx)

    train_data = TextVQADataset(train_ds, split="train")
    val_data = TextVQADataset(val_ds, split="validation")
    test_data = TextVQADataset(test_ds, split="test")

    train_collate = partial(collate_fn, processor=processor)
    eval_collate = partial(collate_fn, processor=processor)

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader
