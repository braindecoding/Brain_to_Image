#!/usr/bin/env python3

"""
Pre-trained CLIP ViT-B/32 for MNIST Digits Transfer Learning

Purpose:
    Use pre-trained OpenAI CLIP ViT-B/32 and fine-tune for MNIST digits
    Much better starting point than training from scratch

Architecture:
    - Image Encoder: Pre-trained ViT-B/32 from OpenAI CLIP
    - Text Encoder: Pre-trained Transformer from OpenAI CLIP  
    - Fine-tuning: Adapt for MNIST 28x28 → 224x224 and digit captions
    - Output: 512-dimensional aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v2.0 (Pre-trained)

Usage:
    model = CLIPPretrainedMNIST()
    image_features, text_features = model(images, text_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Tuple, List
import numpy as np

class CLIPPretrainedMNIST(nn.Module):
    """
    Pre-trained CLIP model adapted for MNIST digits
    Uses OpenAI's pre-trained ViT-B/32 as backbone
    """
    
    def __init__(self, device='cuda', freeze_backbone=False, fine_tune_layers=2):
        super().__init__()
        
        self.device = device
        
        # Load pre-trained CLIP model
        print("📥 Loading pre-trained CLIP ViT-B/32...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Get embedding dimensions
        self.embed_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32
        
        # Image preprocessing for MNIST (28x28 → 224x224)
        self.image_transform = nn.Sequential(
            # Upsample MNIST 28x28 to 224x224
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            # Convert grayscale to RGB (repeat channels)
            # This will be handled in forward pass
        )
        
        # Freeze or fine-tune backbone
        if freeze_backbone:
            print("🔒 Freezing CLIP backbone...")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print(f"🔧 Fine-tuning last {fine_tune_layers} layers...")
            # Freeze most layers, only fine-tune last few
            self._freeze_except_last_layers(fine_tune_layers)
        
        # Additional projection layers for better adaptation
        self.image_adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        self.text_adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        print(f"✅ Pre-trained CLIP loaded with {self.embed_dim}-dim embeddings")
    
    def _freeze_except_last_layers(self, num_layers):
        """Freeze all layers except the last num_layers"""
        # Freeze vision transformer layers except last few
        vision_layers = self.clip_model.visual.transformer.resblocks
        for i, layer in enumerate(vision_layers):
            if i < len(vision_layers) - num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                print(f"  Fine-tuning vision layer {i}")
        
        # Freeze text transformer layers except last few  
        text_layers = self.clip_model.transformer.resblocks
        for i, layer in enumerate(text_layers):
            if i < len(text_layers) - num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                print(f"  Fine-tuning text layer {i}")
    
    def preprocess_images(self, images):
        """
        Preprocess MNIST images for CLIP
        Convert 28x28 grayscale to 224x224 RGB
        """
        # images: (batch_size, 1, 28, 28)
        batch_size = images.shape[0]
        
        # Upsample to 224x224
        images_upsampled = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB by repeating channels
        images_rgb = images_upsampled.repeat(1, 3, 1, 1)  # (batch_size, 3, 224, 224)
        
        # Normalize using CLIP's normalization
        # CLIP expects images normalized with ImageNet stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1)
        
        images_normalized = (images_rgb - mean) / std
        
        return images_normalized
    
    def encode_image(self, images):
        """Encode images using pre-trained CLIP vision encoder"""
        # Preprocess MNIST images for CLIP
        processed_images = self.preprocess_images(images)
        
        # Encode with CLIP vision encoder
        with torch.cuda.amp.autocast():  # Use mixed precision for efficiency
            image_features = self.clip_model.encode_image(processed_images)
        
        # Apply adapter for better MNIST-specific features
        image_features = self.image_adapter(image_features)
        
        # L2 normalize
        image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(self, text_tokens):
        """Encode text using pre-trained CLIP text encoder"""
        # text_tokens should be CLIP-compatible tokens
        with torch.cuda.amp.autocast():  # Use mixed precision for efficiency
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Apply adapter for better digit caption features
        text_features = self.text_adapter(text_features)
        
        # L2 normalize
        text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def forward(self, images, text_tokens):
        """
        Forward pass for contrastive learning
        
        Args:
            images: (batch_size, 1, 28, 28) MNIST images
            text_tokens: (batch_size, 77) CLIP-tokenized text
            
        Returns:
            image_features: (batch_size, embed_dim)
            text_features: (batch_size, embed_dim)
            logit_scale: scalar
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        return image_features, text_features, self.logit_scale.exp()
    
    def compute_similarity(self, image_features, text_features):
        """Compute cosine similarity between image and text features"""
        # Features should already be normalized
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * image_features @ text_features.T
        return similarity

class CLIPTokenizer:
    """
    CLIP-compatible tokenizer for digit captions
    Uses OpenAI's CLIP tokenizer
    """
    
    def __init__(self):
        # Load CLIP tokenizer
        self.tokenizer = clip.tokenize
        
    def encode(self, texts: List[str]):
        """
        Encode list of texts using CLIP tokenizer
        
        Args:
            texts: List of caption strings
            
        Returns:
            tokens: (batch_size, 77) tokenized text
        """
        # CLIP tokenizer expects list of strings
        tokens = self.tokenizer(texts)  # Returns (batch_size, 77)
        return tokens
    
    def encode_batch(self, captions: List[str]):
        """Encode batch of captions"""
        return self.encode(captions)

def test_clip_pretrained():
    """Test function for pre-trained CLIP model"""
    print("Testing CLIPPretrainedMNIST...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize model
        model = CLIPPretrainedMNIST(device=device, freeze_backbone=False)
        
        # Test data
        batch_size = 4
        images = torch.randn(batch_size, 1, 28, 28).to(device)  # MNIST format
        
        # Test captions
        captions = [
            "A handwritten digit zero",
            "A handwritten digit one", 
            "A handwritten digit two",
            "A handwritten digit three"
        ]
        
        # Tokenize captions
        tokenizer = CLIPTokenizer()
        text_tokens = tokenizer.encode(captions).to(device)
        
        print(f"Image shape: {images.shape}")
        print(f"Text tokens shape: {text_tokens.shape}")
        
        # Forward pass
        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, text_tokens)
            
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Logit scale: {logit_scale.item():.4f}")
        
        # Test similarity
        similarity = model.compute_similarity(image_features, text_features)
        print(f"Similarity matrix shape: {similarity.shape}")
        print(f"Similarity values: {similarity.diag()}")  # Diagonal should be highest
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n✅ Pre-trained CLIP test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_pretrained()
