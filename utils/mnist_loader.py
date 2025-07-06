#!/usr/bin/env python3

"""
MNIST Image Loader Utility

Purpose:
    Load MNIST images corresponding to EEG labels
    Used in CLIP training pipeline for image-text alignment

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0

Usage:
    loader = MNISTLoader('Datasets/MNIST_dataset')
    images = loader.get_images_for_labels([0, 1, 2, 3, 4])
    image = loader.get_random_image_for_digit(5)
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from typing import List, Union, Tuple, Optional
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MNISTLoader:
    """
    MNIST Image Loader for CLIP Training
    
    Loads MNIST images corresponding to digit labels
    """
    
    def __init__(self, data_dir: str = "Datasets/MNIST_dataset", download: bool = True):
        """
        Initialize MNIST loader
        
        Args:
            data_dir: Directory to store/load MNIST data
            download: Whether to download MNIST if not exists
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Load MNIST datasets
        print(f"📂 Loading MNIST data from: {data_dir}")
        try:
            self.train_dataset = datasets.MNIST(
                root=data_dir, 
                train=True, 
                download=download, 
                transform=self.transform
            )
            
            self.test_dataset = datasets.MNIST(
                root=data_dir, 
                train=False, 
                download=download, 
                transform=self.transform
            )
            
            print(f"✅ MNIST loaded: {len(self.train_dataset)} train, {len(self.test_dataset)} test")
            
            # Create label-to-indices mapping for efficient lookup
            self._create_label_mappings()

            # Create prototype/representative images for consistent reconstruction
            self._create_prototype_mappings()
            
        except Exception as e:
            print(f"❌ Error loading MNIST: {e}")
            raise
    
    def _create_label_mappings(self):
        """Create mappings from labels to dataset indices for fast lookup"""
        print("🔍 Creating label-to-indices mappings...")
        
        # Train set mapping
        self.train_label_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.train_dataset):
            self.train_label_to_indices[label].append(idx)
        
        # Test set mapping
        self.test_label_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.test_dataset):
            self.test_label_to_indices[label].append(idx)
        
        # Print statistics
        for digit in range(10):
            train_count = len(self.train_label_to_indices[digit])
            test_count = len(self.test_label_to_indices[digit])
            print(f"  Digit {digit}: {train_count} train, {test_count} test images")

    def _create_prototype_mappings(self):
        """Create prototype/representative images for consistent reconstruction"""
        print("🎯 Creating prototype mappings for consistent reconstruction...")

        # For each digit, select a representative image
        # Strategy: Use the first image in the sorted list for consistency
        self.train_prototypes = {}
        self.test_prototypes = {}

        for digit in range(10):
            # Train prototypes - use first available index for consistency
            train_indices = sorted(self.train_label_to_indices[digit])
            if train_indices:
                self.train_prototypes[digit] = train_indices[0]

            # Test prototypes - use first available index for consistency
            test_indices = sorted(self.test_label_to_indices[digit])
            if test_indices:
                self.test_prototypes[digit] = test_indices[0]

        print("✅ Prototype mappings created for consistent reconstruction")

        # Print prototype indices for reference
        print("📋 Prototype indices:")
        for digit in range(10):
            train_proto = self.train_prototypes.get(digit, 'N/A')
            test_proto = self.test_prototypes.get(digit, 'N/A')
            print(f"  Digit {digit}: train[{train_proto}], test[{test_proto}]")
    
    def get_random_image_for_digit(self, digit: int, split: str = 'train') -> torch.Tensor:
        """
        Get a random MNIST image for specified digit
        
        Args:
            digit: Digit class (0-9)
            split: Dataset split ('train' or 'test')
            
        Returns:
            image: MNIST image tensor (1, 28, 28)
        """
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be 0-9, got {digit}")
        
        # Select dataset and mapping
        if split == 'train':
            dataset = self.train_dataset
            label_mapping = self.train_label_to_indices
        elif split == 'test':
            dataset = self.test_dataset
            label_mapping = self.test_label_to_indices
        else:
            raise ValueError(f"Split must be 'train' or 'test', got {split}")
        
        # Get random index for the digit
        available_indices = label_mapping[digit]
        if not available_indices:
            raise ValueError(f"No images found for digit {digit} in {split} set")
        
        random_idx = random.choice(available_indices)
        image, label = dataset[random_idx]
        
        return image

    def get_prototype_image_for_digit(self, digit: int, split: str = 'train') -> torch.Tensor:
        """
        Get the CONSISTENT prototype image for specified digit

        Args:
            digit: Digit class (0-9)
            split: Dataset split ('train' or 'test')

        Returns:
            image: Consistent MNIST prototype image tensor (1, 28, 28)
        """
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be 0-9, got {digit}")

        # Select dataset and prototype mapping
        if split == 'train':
            dataset = self.train_dataset
            prototype_mapping = self.train_prototypes
        elif split == 'test':
            dataset = self.test_dataset
            prototype_mapping = self.test_prototypes
        else:
            raise ValueError(f"Split must be 'train' or 'test', got {split}")

        # Get prototype index
        if digit not in prototype_mapping:
            raise ValueError(f"No prototype found for digit {digit} in {split} set")

        prototype_idx = prototype_mapping[digit]
        image, _ = dataset[prototype_idx]

        return image
    
    def get_images_for_labels(self, labels: List[int], split: str = 'train', mode: str = 'random') -> torch.Tensor:
        """
        Get MNIST images for a batch of labels

        Args:
            labels: List of digit labels (0-9)
            split: Dataset split ('train' or 'test')
            mode: Image selection mode ('random' or 'prototype')

        Returns:
            images: Batch of MNIST images (batch_size, 1, 28, 28)
        """
        images = []
        for label in labels:
            if mode == 'prototype':
                image = self.get_prototype_image_for_digit(label, split)
            else:  # mode == 'random'
                image = self.get_random_image_for_digit(label, split)
            images.append(image)

        return torch.stack(images)

    def get_prototype_images_for_labels(self, labels: List[int], split: str = 'train') -> torch.Tensor:
        """
        Get CONSISTENT prototype MNIST images for a batch of labels

        Args:
            labels: List of digit labels (0-9)
            split: Dataset split ('train' or 'test')

        Returns:
            images: Batch of consistent prototype MNIST images (batch_size, 1, 28, 28)
        """
        return self.get_images_for_labels(labels, split, mode='prototype')

    def get_all_prototype_images(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all 10 prototype images (one for each digit 0-9)

        Args:
            split: Dataset split ('train' or 'test')

        Returns:
            images: All prototype images (10, 1, 28, 28)
            labels: Corresponding labels [0, 1, 2, ..., 9] (10,)
        """
        labels = list(range(10))
        images = self.get_prototype_images_for_labels(labels, split)
        return images, torch.tensor(labels)

    def create_prototype_gallery(self, split: str = 'train') -> torch.Tensor:
        """
        Create a gallery of all prototype images for visualization

        Args:
            split: Dataset split ('train' or 'test')

        Returns:
            gallery: Gallery image tensor for visualization
        """
        prototype_images, _ = self.get_all_prototype_images(split)

        # Arrange in 2x5 grid for visualization
        # Reshape from (10, 1, 28, 28) to (2, 5, 28, 28) then combine
        gallery_rows = []
        for row in range(2):
            row_images = prototype_images[row*5:(row+1)*5]  # 5 images per row
            row_combined = torch.cat([img.squeeze(0) for img in row_images], dim=1)  # Concatenate horizontally
            gallery_rows.append(row_combined)

        # Combine rows vertically
        gallery = torch.cat(gallery_rows, dim=0)  # Shape: (56, 140) = (2*28, 5*28)

        return gallery
    
    def get_balanced_batch(self, batch_size: int, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a balanced batch with equal representation of all digits
        
        Args:
            batch_size: Total batch size (should be divisible by 10)
            split: Dataset split ('train' or 'test')
            
        Returns:
            images: Batch of images (batch_size, 1, 28, 28)
            labels: Corresponding labels (batch_size,)
        """
        if batch_size % 10 != 0:
            print(f"⚠️ Warning: batch_size {batch_size} not divisible by 10, using {batch_size//10*10}")
            batch_size = batch_size // 10 * 10
        
        samples_per_digit = batch_size // 10
        
        images = []
        labels = []
        
        for digit in range(10):
            for _ in range(samples_per_digit):
                image = self.get_random_image_for_digit(digit, split)
                images.append(image)
                labels.append(digit)
        
        # Shuffle the batch
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)
        
        return torch.stack(images), torch.tensor(labels)
    
    def get_specific_images(self, digit: int, count: int, split: str = 'train') -> torch.Tensor:
        """
        Get specific number of images for a digit
        
        Args:
            digit: Digit class (0-9)
            count: Number of images to get
            split: Dataset split ('train' or 'test')
            
        Returns:
            images: Images tensor (count, 1, 28, 28)
        """
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be 0-9, got {digit}")
        
        # Select dataset and mapping
        if split == 'train':
            dataset = self.train_dataset
            label_mapping = self.train_label_to_indices
        else:
            dataset = self.test_dataset
            label_mapping = self.test_label_to_indices
        
        available_indices = label_mapping[digit]
        if len(available_indices) < count:
            print(f"⚠️ Warning: Only {len(available_indices)} images available for digit {digit}, requested {count}")
            count = len(available_indices)
        
        # Sample random indices
        selected_indices = random.sample(available_indices, count)
        
        images = []
        for idx in selected_indices:
            image, _ = dataset[idx]
            images.append(image)
        
        return torch.stack(images)
    
    def get_dataset_stats(self) -> dict:
        """Get statistics about the MNIST dataset"""
        stats = {
            'train_total': len(self.train_dataset),
            'test_total': len(self.test_dataset),
            'train_per_digit': {},
            'test_per_digit': {}
        }
        
        for digit in range(10):
            stats['train_per_digit'][digit] = len(self.train_label_to_indices[digit])
            stats['test_per_digit'][digit] = len(self.test_label_to_indices[digit])
        
        return stats


def test_mnist_loader():
    """Test function for MNIST loader"""
    print("Testing MNISTLoader...")
    
    try:
        # Initialize loader
        loader = MNISTLoader()
        
        # Test single image
        print("\n=== Single Image Test ===")
        image = loader.get_random_image_for_digit(5)
        print(f"Single image shape: {image.shape}")
        print(f"Image range: {image.min():.3f} to {image.max():.3f}")
        
        # Test batch images
        print("\n=== Batch Images Test ===")
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_images = loader.get_images_for_labels(labels)
        print(f"Batch images shape: {batch_images.shape}")
        
        # Test balanced batch
        print("\n=== Balanced Batch Test ===")
        bal_images, bal_labels = loader.get_balanced_batch(20)
        print(f"Balanced batch - Images: {bal_images.shape}, Labels: {bal_labels.shape}")
        print(f"Label distribution: {torch.bincount(bal_labels)}")
        
        # Test specific images
        print("\n=== Specific Images Test ===")
        specific_images = loader.get_specific_images(7, 5)
        print(f"Specific images shape: {specific_images.shape}")

        # Test prototype images (NEW!)
        print("\n=== Prototype Images Test ===")
        proto_image = loader.get_prototype_image_for_digit(5)
        print(f"Prototype image shape: {proto_image.shape}")

        # Test prototype batch
        proto_labels = [0, 1, 2, 3, 4]
        proto_batch = loader.get_prototype_images_for_labels(proto_labels)
        print(f"Prototype batch shape: {proto_batch.shape}")

        # Test all prototypes
        all_protos, all_labels = loader.get_all_prototype_images()
        print(f"All prototypes shape: {all_protos.shape}")
        print(f"All labels: {all_labels}")

        # Test consistency - same digit should give same image
        print("\n=== Consistency Test ===")
        img1 = loader.get_prototype_image_for_digit(3)
        img2 = loader.get_prototype_image_for_digit(3)
        is_same = torch.equal(img1, img2)
        print(f"Same digit gives same image: {is_same}")

        # Test gallery creation
        print("\n=== Gallery Test ===")
        gallery = loader.create_prototype_gallery()
        print(f"Gallery shape: {gallery.shape} (should be 56x140 for 2x5 grid)")

        # Test dataset stats
        print("\n=== Dataset Statistics ===")
        stats = loader.get_dataset_stats()
        print(f"Train total: {stats['train_total']}")
        print(f"Test total: {stats['test_total']}")
        print("Train per digit:", stats['train_per_digit'])

        print("\n✅ MNIST loader test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_mnist_loader()
