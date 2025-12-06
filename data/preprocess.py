"""
Data preprocessing utilities for TextVQA dataset.
Handles data loading, tokenization, and preparation for model input.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class TextVQADataset(Dataset):
    """
    Custom Dataset class for TextVQA with Qwen2.5-VL model.
    """
    
    def __init__(
        self,
        dataset,
        processor,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Args:
            dataset: HuggingFace dataset object
            processor: Qwen2.5-VL processor
            max_length: Maximum sequence length
            split: Dataset split (train/validation/test)
        """
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.split = split
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Returns:
            Dictionary containing processed inputs and targets
        """
        example = self.dataset[idx]
        
        # Get image
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        
        # Get question
        question = example['question']
        
        # Get answers (TextVQA has multiple acceptable answers)
        answers = example.get('answers', [])
        if isinstance(answers, list) and len(answers) > 0:
            # Use the first answer as target during training
            answer = answers[0] if isinstance(answers[0], str) else answers[0]
        else:
            answer = str(answers)
        
        # Format prompt for Qwen2.5-VL
        prompt = f"Question: {question}\nAnswer:"
        
        # Process inputs
        try:
            # For Qwen2.5-VL, we need to prepare conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process image and text
            # Don't use max_length truncation for vision models - images have variable token counts
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True  # Dynamic padding instead of max_length
            )
            
            # Process answer for training
            if self.split == "train":
                answer_inputs = self.processor.tokenizer(
                    answer,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True
                )
                labels = answer_inputs["input_ids"].squeeze(0)
            else:
                labels = None
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs.get("pixel_values", inputs.get("image_grid_thw")).squeeze(0) if "pixel_values" in inputs or "image_grid_thw" in inputs else None,
                "labels": labels,
                "question": question,
                "answers": answers,
                "question_id": example.get('question_id', idx),
                "image_id": example.get('image_id', '')
            }
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return a dummy example in case of error
            return self._get_dummy_example()
    
    def _get_dummy_example(self):
        """Return a dummy example in case of processing errors."""
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "pixel_values": None,
            "labels": None,
            "question": "",
            "answers": [],
            "question_id": -1,
            "image_id": ""
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching of TextVQA data with variable-length sequences.
    
    Args:
        batch: List of examples from TextVQADataset
        
    Returns:
        Batched dictionary
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Pad input_ids and attention_mask to max_len
    input_ids_list = []
    attention_mask_list = []
    
    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        
        # Calculate padding needed
        pad_len = max_len - input_ids.size(0)
        
        if pad_len > 0:
            # Pad with zeros (pad token id is usually 0)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    
    # Stack the padded tensors
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    
    # Handle pixel values (may be None for some examples)
    # For Qwen2.5-VL, pixel values have shape [num_patches, hidden_dim]
    # Pad to max_patches in batch
    pixel_values = None
    if batch[0]["pixel_values"] is not None:
        pv_list = [item["pixel_values"] for item in batch]
        
        # Check if all have same shape
        shapes = [pv.shape for pv in pv_list]
        if len(set(shapes)) == 1:
            # All same shape - can stack directly
            pixel_values = torch.stack(pv_list)
        else:
            # Different shapes - pad to max size
            max_patches = max(pv.shape[0] for pv in pv_list)
            hidden_dim = pv_list[0].shape[1]
            
            padded_pv_list = []
            for pv in pv_list:
                if pv.shape[0] < max_patches:
                    # Pad with zeros
                    pad_size = max_patches - pv.shape[0]
                    padding = torch.zeros(pad_size, hidden_dim, dtype=pv.dtype)
                    padded_pv = torch.cat([pv, padding], dim=0)
                else:
                    padded_pv = pv
                padded_pv_list.append(padded_pv)
            
            pixel_values = torch.stack(padded_pv_list)
    
    # Handle labels
    labels = None
    if batch[0]["labels"] is not None:
        labels = torch.stack([item["labels"] for item in batch])
    
    # Keep metadata as lists
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    question_ids = [item["question_id"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
        "questions": questions,
        "answers": answers,
        "question_ids": question_ids,
        "image_ids": image_ids
    }


def load_textvqa_data(
    data_dir: str = "./textvqa_data",
    split: str = "train",
    use_hf_direct: bool = True
) -> Dataset:
    """
    Load TextVQA dataset from local disk or Hugging Face.
    
    Args:
        data_dir: Directory containing saved dataset
        split: Dataset split to load
        use_hf_direct: If True, load directly from HuggingFace; else from disk
        
    Returns:
        HuggingFace Dataset object
    """
    if use_hf_direct:
        print(f"Loading {split} split directly from Hugging Face...")
        dataset = load_dataset("lmms-lab/textvqa", split=split, trust_remote_code=True)
    else:
        split_path = Path(data_dir) / split / "data"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {split_path}. "
                "Please run download_data.py first or use use_hf_direct=True"
            )
        print(f"Loading {split} split from {split_path}...")
        dataset = load_from_disk(str(split_path))
    
    print(f"Loaded {len(dataset)} examples from {split} split")
    return dataset


def create_dataloaders(
    processor,
    data_dir: str = "./textvqa_data",
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    max_length: int = 512,
    use_hf_direct: bool = True,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        processor: Model processor
        data_dir: Directory containing dataset
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        max_length: Maximum sequence length
        use_hf_direct: Load from HuggingFace directly
        subset_size: Use a subset of training data (for ablation studies)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_dataset = load_textvqa_data(data_dir, "train", use_hf_direct)
    val_dataset = load_textvqa_data(data_dir, "validation", use_hf_direct)
    test_dataset = load_textvqa_data(data_dir, "test", use_hf_direct)
    
    # Create subset if specified (for ablation studies)
    if subset_size is not None and subset_size < len(train_dataset):
        print(f"Using subset of {subset_size} training examples")
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = train_dataset.select(indices)
    
    # Create Dataset objects
    train_data = TextVQADataset(train_dataset, processor, max_length, "train")
    val_data = TextVQADataset(val_dataset, processor, max_length, "validation")
    test_data = TextVQADataset(test_dataset, processor, max_length, "test")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def prepare_data_for_ablation(
    processor,
    data_dir: str = "./textvqa_data",
    fraction: float = 1.0,
    batch_size: int = 8,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data for ablation studies with different data fractions.
    
    Args:
        processor: Model processor
        data_dir: Directory containing dataset
        fraction: Fraction of training data to use (0.0 to 1.0)
        batch_size: Batch size
        **kwargs: Additional arguments for create_dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = load_textvqa_data(data_dir, "train", kwargs.get('use_hf_direct', True))
    subset_size = int(len(train_dataset) * fraction)
    
    train_loader, val_loader, _ = create_dataloaders(
        processor=processor,
        data_dir=data_dir,
        train_batch_size=batch_size,
        subset_size=subset_size,
        **kwargs
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    from transformers import Qwen2VLProcessor
    
    print("Testing data loading and preprocessing...")
    
    # Initialize processor
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Load a small subset
    dataset = load_textvqa_data(split="validation", use_hf_direct=True)
    
    # Create dataset
    test_dataset = TextVQADataset(dataset, processor, split="validation")
    
    # Get a sample
    sample = test_dataset[0]
    print("\nSample data:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor of shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Data preprocessing test completed successfully!")
