from abc import ABC, abstractmethod
import torch
from PIL import Image

class BaseModel(ABC):
    def __init__(self, model_path, device="cuda:3" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass

    @abstractmethod
    def generate_answer(self, image: Image.Image, question: str, max_new_tokens=20):
        """
        Generate an answer for the given image and question.
        
        Args:
            image (PIL.Image): Input image.
            question (str): Input question.
            max_new_tokens (int): Maximum number of tokens to generate.
            
        Returns:
            str: Generated answer.
        """
        pass

    def fine_tune(self, train_dataset, output_dir, epochs=1, learning_rate=2e-5):
        """
        Fine-tune the model using LoRA.
        
        Args:
            train_dataset: Training dataset.
            output_dir: Directory to save the fine-tuned model.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
        """
        raise NotImplementedError("Fine-tuning not implemented for this model yet.")
