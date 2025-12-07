import os
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset

class TextVQADataset(Dataset):
    def __init__(self, split="train", cache_dir=None):
        """
        Args:
            split (str): One of "train", "validation", "test".
            cache_dir (str, optional): Directory to cache the dataset.
        """
        self.split = split
        print(f"Loading TextVQA dataset split: {split}...")
        self.dataset = load_dataset("lmms-lab/textvqa", split=split, cache_dir=cache_dir)
        print(f"Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # The dataset from huggingface usually has 'image', 'question', 'answers' (list), 'image_id'
        image = item['image']
        question = item['question']
        image_id = item['image_id']
        
        # 'answers' might not be present in test split or might be in a specific format
        answers = item.get('answers', [])
        ocr_tokens = item.get('ocr_tokens', [])
        
        return {
            "image": image,
            "question": question,
            "answers": answers,
            "image_id": image_id,
            "ocr_tokens": ocr_tokens
        }

if __name__ == "__main__":
    # Simple test
    ds = TextVQADataset(split="validation")
    sample = ds[0]
    print("Sample 0:", sample)
    print("Image size:", sample['image'].size)
