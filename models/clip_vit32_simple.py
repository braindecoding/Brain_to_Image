#!/usr/bin/env python3

"""
Simple CLIP with ViT-B/32 for Contrastive Learning Only

Purpose:
    Use OpenAI's pre-trained ViT-B/32 for pure contrastive learning
    No multi-task complexity, just image-text alignment
    Much simpler and more stable approach

Architecture:
    - Image Encoder: Pre-trained ViT-B/32 from OpenAI CLIP
    - Text Encoder: Pre-trained Transformer from OpenAI CLIP
    - Training: Pure contrastive learning (InfoNCE loss)
    - Output: 512-dimensional aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v5.0 (Simple Contrastive Only)

Usage:
    model = CLIPViTSimple()
    image_features, text_features = model(images, text_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Tuple, List
import numpy as np

class CLIPViTSimple(nn.Module):
    """
    Simple CLIP model using pre-trained ViT-B/32 for contrastive learning only
    No classification head, no multi-task complexity
    """
    
    def __init__(self, device='cuda', freeze_backbone=False, 
                 temperature: float = 0.07, adaptation_layers: bool = True):
        super().__init__()
        
        self.device = device
        
        # Load pre-trained CLIP model
        print("📥 Loading pre-trained CLIP ViT-B/32...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, 
                                                     download_root="/raid/data/m33218012/clip_cache")
        
        # Get embedding dimensions
        self.embed_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32
        
        # Freeze or fine-tune backbone
        if freeze_backbone:
            print("🔒 Freezing CLIP backbone...")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("🔧 Fine-tuning CLIP backbone...")
        
        # Optional adaptation layers for better MNIST-specific features
        if adaptation_layers:
            print("🔧 Adding adaptation layers...")
            self.image_adapter = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)
            
            self.text_adapter = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)
        else:
            print("🚀 No adaptation layers - pure CLIP...")
            self.image_adapter = None
            self.text_adapter = None
        
        # Learnable temperature parameter - CONSERVATIVE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature)).to(device)
        
        print(f"✅ Simple CLIP loaded with {self.embed_dim}-dim embeddings")
        print(f"   Adaptation layers: {adaptation_layers}")
        print(f"   Frozen backbone: {freeze_backbone}")
    
    def preprocess_images(self, images):
        """
        Preprocess MNIST images for CLIP ViT-B/32
        Convert 28x28 grayscale to 224x224 RGB
        """
        # images: (batch_size, 1, 28, 28)
        batch_size = images.shape[0]
        
        # Upsample to 224x224
        images_upsampled = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB by repeating channels
        images_rgb = images_upsampled.repeat(1, 3, 1, 1)  # (batch_size, 3, 224, 224)
        
        # Normalize using CLIP's normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1)
        
        images_normalized = (images_rgb - mean) / std
        
        return images_normalized
    
    def encode_image(self, images):
        """Encode images using CLIP vision encoder"""
        # Preprocess MNIST images for CLIP
        processed_images = self.preprocess_images(images)
        
        # Encode with CLIP vision encoder
        image_features = self.clip_model.encode_image(processed_images).float()
        
        # Apply adaptation if available
        if self.image_adapter is not None:
            image_features = self.image_adapter(image_features)
        
        # L2 normalize
        image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(self, text_tokens):
        """Encode text using CLIP text encoder"""
        # Encode with CLIP text encoder
        text_features = self.clip_model.encode_text(text_tokens).float()
        
        # Apply adaptation if available
        if self.text_adapter is not None:
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
        
        # Ensure same dtype
        image_features = image_features.float()
        text_features = text_features.float()
        logit_scale = logit_scale.float()
        
        similarity = logit_scale * image_features @ text_features.T
        return similarity

def test_clip_simple():
    """Test function for simple CLIP model"""
    print("Testing Simple CLIP ViT-B/32...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test both configurations
        configs = [
            {"adaptation_layers": False, "freeze_backbone": True, "name": "Pure CLIP (Frozen)"},
            {"adaptation_layers": True, "freeze_backbone": False, "name": "CLIP + Adaptation (Fine-tuned)"},
        ]
        
        for config in configs:
            print(f"\n🧪 Testing: {config['name']}")
            print("-" * 50)
            
            # Initialize model
            model = CLIPViTSimple(
                device=device,
                freeze_backbone=config["freeze_backbone"],
                adaptation_layers=config["adaptation_layers"]
            )
            
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
            
            # Tokenize captions using CLIP tokenizer
            text_tokens = clip.tokenize(captions).to(device)
            
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
            print(f"Similarity diagonal: {similarity.diag()}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            print(f"✅ {config['name']} test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_simple()
