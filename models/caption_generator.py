#!/usr/bin/env python3

"""
Template-based Caption Generator for MNIST Digits

Purpose:
    Generate descriptive captions for MNIST digit predictions
    Used in CLIP-style training for cross-modal alignment

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0

Usage:
    generator = DigitCaptionGenerator()
    caption = generator.generate_caption(digit_prediction)
    captions = generator.generate_batch_captions(digit_predictions)
"""

import torch
import numpy as np
from typing import List, Union

class DigitCaptionGenerator:
    """
    Template-based caption generator for MNIST digits
    
    Generates descriptive captions in the format:
    "A handwritten digit {number}"
    """
    
    def __init__(self):
        """Initialize caption templates"""
        self.templates = [
            "A handwritten digit zero",
            "A handwritten digit one", 
            "A handwritten digit two",
            "A handwritten digit three",
            "A handwritten digit four",
            "A handwritten digit five",
            "A handwritten digit six",
            "A handwritten digit seven",
            "A handwritten digit eight",
            "A handwritten digit nine"
        ]
        
        # Alternative templates for variety (optional)
        self.alternative_templates = [
            "The digit {digit}",
            "Number {digit}",
            "A written {digit}",
            "Handwritten {digit}",
            "The number {digit}"
        ]
        
        self.digit_names = [
            "zero", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine"
        ]
    
    def generate_caption(self, digit: Union[int, torch.Tensor, np.ndarray]) -> str:
        """
        Generate caption for a single digit prediction
        
        Args:
            digit: Digit prediction (0-9) as int, tensor, or array
            
        Returns:
            caption: Descriptive caption string
        """
        # Convert to int if tensor or array
        if isinstance(digit, (torch.Tensor, np.ndarray)):
            digit = int(digit.item() if hasattr(digit, 'item') else digit)
        
        # Validate digit range
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be between 0-9, got {digit}")
        
        return self.templates[digit]
    
    def generate_batch_captions(self, digits: Union[List[int], torch.Tensor, np.ndarray]) -> List[str]:
        """
        Generate captions for batch of digit predictions
        
        Args:
            digits: Batch of digit predictions
            
        Returns:
            captions: List of descriptive caption strings
        """
        # Convert tensor/array to list
        if isinstance(digits, (torch.Tensor, np.ndarray)):
            if isinstance(digits, torch.Tensor):
                digits = digits.cpu().numpy()
            digits = digits.tolist()
        
        return [self.generate_caption(digit) for digit in digits]
    
    def generate_alternative_caption(self, digit: Union[int, torch.Tensor, np.ndarray], 
                                   template_idx: int = 0) -> str:
        """
        Generate alternative caption using different templates
        
        Args:
            digit: Digit prediction (0-9)
            template_idx: Index of alternative template to use
            
        Returns:
            caption: Alternative descriptive caption string
        """
        # Convert to int if tensor or array
        if isinstance(digit, (torch.Tensor, np.ndarray)):
            digit = int(digit.item() if hasattr(digit, 'item') else digit)
        
        # Validate digit range
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be between 0-9, got {digit}")
        
        # Validate template index
        if not (0 <= template_idx < len(self.alternative_templates)):
            template_idx = 0
        
        template = self.alternative_templates[template_idx]
        digit_name = self.digit_names[digit]
        
        return template.format(digit=digit_name)
    
    def get_all_captions(self) -> List[str]:
        """
        Get all possible captions for digits 0-9
        
        Returns:
            captions: List of all 10 digit captions
        """
        return self.templates.copy()
    
    def get_caption_for_class(self, class_idx: int) -> str:
        """
        Get caption for specific class index (same as generate_caption)
        
        Args:
            class_idx: Class index (0-9)
            
        Returns:
            caption: Caption for the class
        """
        return self.generate_caption(class_idx)


def test_caption_generator():
    """Test function for caption generator"""
    print("Testing DigitCaptionGenerator...")
    
    generator = DigitCaptionGenerator()
    
    # Test single digit
    print("\n=== Single Digit Tests ===")
    for digit in range(10):
        caption = generator.generate_caption(digit)
        print(f"Digit {digit}: {caption}")
    
    # Test batch
    print("\n=== Batch Test ===")
    batch_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_captions = generator.generate_batch_captions(batch_digits)
    for digit, caption in zip(batch_digits, batch_captions):
        print(f"Batch {digit}: {caption}")
    
    # Test tensor input
    print("\n=== Tensor Input Test ===")
    tensor_digits = torch.tensor([3, 7, 1, 9])
    tensor_captions = generator.generate_batch_captions(tensor_digits)
    for digit, caption in zip(tensor_digits, tensor_captions):
        print(f"Tensor {digit}: {caption}")
    
    # Test alternative templates
    print("\n=== Alternative Templates Test ===")
    for i in range(len(generator.alternative_templates)):
        alt_caption = generator.generate_alternative_caption(5, i)
        print(f"Alt template {i}: {alt_caption}")
    
    print("\n✅ Caption generator test completed!")


if __name__ == "__main__":
    test_caption_generator()
