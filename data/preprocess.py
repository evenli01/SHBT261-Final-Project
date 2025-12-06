"""
Data preprocessing utilities for TextVQA dataset.
Loads local HF datasets, processes images + questions, builds batches.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# ============================================================
#                   Load Local TextVQA Dataset 
# ============================================================

def load_textvqa_data(
    data_dir: str,
    split: str = "train",
    use_hf_direct: bool = False
):
    """
    Load TextVQA dataset **ONLY from local disk**.
    """
    if use_hf_direct:
        raise ValueError(
            "Direct HF loading is disabled — dataset must be loaded from local disk."
        )

    split_path = Path(data_dir) / split / "data"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {split_path} — "
            f"ensure download_data.py stored local HF dataset correctly."
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
    TextVQA dataset for Qwen2.5-VL. Ensures pixel_values ALWAYS exist.
    """

    def __init__(
        self,
        dataset,
        processor,
        max_length: int = 512,
        split: str = "train"
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Load image (path → PIL)
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        question = example["question"]
        answers = example.get("answers", [])

        # Choose one answer for training; keep list for eval
        answer = answers[0] if isinstance(answers, list) and len(answers) > 0 else ""

        # Qwen VL chat-format input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process both text+image
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # ========== FIX: Guarantee pixel_values exists ==============
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].squeeze(0)
        elif "image_grid_thw" in inputs:
            pixel_values = inputs["image_grid_thw"].squeeze(0)
        else:
            raise ValueError(
                f"Processor returned no image tensor. Keys={inputs.keys()}"
            )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # For training, prepare label tokens
        if self.split == "train":
            label_ids = self.processor.tokenizer(
                answer,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        else:
            label_ids = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": label_ids,
            "question": question,
            "answers": answers,
            "question_id": example.get("question_id", idx),
            "image_id": example.get("image_id", "")
        }


# ============================================================
#                       Collate Function
# ============================================================

def collate_fn(batch: List[Dict]):
    """
    Batch input_ids, attention_mask, pixel_values (always padded).
    """

    # pad text
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_masks = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]

        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

        input_ids.append(ids)
        attention_masks.append(mask)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    # pixel values: simple batch stack (processor resizes all images uniformly)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    labels = (
        torch.stack([item["labels"] for item in batch])
        if batch[0]["labels"] is not None else None
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "pixel_values": pixel_values,
        "labels": labels,
        "questions": [b["question"] for b in batch],
        "answers": [b["answers"] for b in batch],
        "question_ids": [b["question_id"] for b in batch],
        "image_ids": [b["image_id"] for b in batch]
    }


# ============================================================
#               Create train/val/test dataloaders
# ============================================================

def create_dataloaders(
    processor,
    data_dir: str,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 4,
    max_length: int = 512,
    subset_size: Optional[int] = None
):
    train_ds = load_textvqa_data(data_dir, "train")
    val_ds = load_textvqa_data(data_dir, "validation")
    test_ds = load_textvqa_data(data_dir, "test")

    if subset_size and subset_size < len(train_ds):
        idx = np.random.choice(len(train_ds), subset_size, replace=False)
        train_ds = train_ds.select(idx)

    train_data = TextVQADataset(train_ds, processor, split="train")
    val_data = TextVQADataset(val_ds, processor, split="validation")
    test_data = TextVQADataset(test_ds, processor, split="test")

    train_loader = DataLoader(train_data, batch_size=train_batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)

    val_loader = DataLoader(val_data, batch_size=eval_batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers)

    test_loader = DataLoader(test_data, batch_size=eval_batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=num_workers)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    return train_loader, val_loader, test_loader
