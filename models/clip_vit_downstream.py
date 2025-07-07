#!/usr/bin/env python3

"""
CLIP with ViT-base-MNIST as Downstream Task

Purpose:
    Use pre-trained ViT-base-MNIST as a proper downstream task for image classification
    Then adapt it for CLIP-style contrastive learning with text

Architecture:
    - Image Encoder: ViTForImageClassification (pre-trained on MNIST) → feature extraction
    - Text Encoder: CLIP text encoder for digit captions
    - Training: Contrastive learning for image-text alignment
    - Output: 512-dimensional aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v4.0 (Downstream Task Approach)

Usage:
    model = CLIPViTDownstream()
    image_features, text_features = model(images, text_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import ViTImageProcessor, ViTForImageClassification
from typing import Tuple, List
import numpy as np

class CLIPViTDownstream(nn.Module):
    """
    CLIP model using ViT-base-MNIST as downstream task
    Uses ViTForImageClassification (properly trained) + CLIP text encoder
    """
    
    def __init__(self, 
                 # Pre-trained ViT model
                 model_name: str = "farleyknight-org-username/vit-base-mnist",
                 # Shared params
                 output_dim: int = 512,
                 temperature: float = 0.07,
                 freeze_vit: bool = False,
                 device: str = 'cuda'):
        super().__init__()
        
        self.output_dim = output_dim
        self.model_name = model_name
        self.device = device
        
        # Load pre-trained ViT-base-MNIST for classification (downstream task)
        print(f"📥 Loading ViT-base-MNIST for downstream task: {model_name}")
        self.vit_processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit_classifier = ViTForImageClassification.from_pretrained(model_name).to(device)
        
        # Get ViT hidden size from the classifier
        self.vit_hidden_size = self.vit_classifier.config.hidden_size  # Usually 768
        
        # Freeze ViT if requested
        if freeze_vit:
            print("🔒 Freezing ViT-base-MNIST classifier...")
            for param in self.vit_classifier.parameters():
                param.requires_grad = False
        else:
            print("🔧 Fine-tuning ViT-base-MNIST classifier...")
        
        # Load CLIP text encoder
        print("📥 Loading CLIP text encoder...")
        clip_model, _ = clip.load("ViT-B/32", device=device, download_root="/raid/data/m33218012/clip_cache")
        self.text_encoder = clip_model.transformer
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        
        # Image projection from ViT hidden size to target embedding dimension
        self.image_projection = nn.Sequential(
            nn.Linear(self.vit_hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        ).to(device)
        
        # Learnable temperature parameter - CONSERVATIVE for stability
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.5)).to(device)  # Fixed conservative value
        
        print(f"✅ CLIP Downstream Task loaded")
        print(f"   ViT hidden size: {self.vit_hidden_size}")
        print(f"   Output dimension: {output_dim}")
        print(f"   Frozen ViT: {freeze_vit}")
    
    def preprocess_images(self, images):
        """
        Preprocess MNIST images for ViT-base-MNIST classifier
        Convert tensor format to format expected by ViT processor
        """
        # images: (batch_size, 1, 28, 28)
        batch_size = images.shape[0]
        
        # Convert to 3-channel RGB (ViT expects RGB)
        if images.shape[1] == 1:
            images_rgb = images.repeat(1, 3, 1, 1)  # (batch_size, 3, 28, 28)
        else:
            images_rgb = images
        
        # Normalize to [0, 1] range
        images_rgb = torch.clamp(images_rgb, 0, 1)
        if images_rgb.min() < 0 or images_rgb.max() > 1:
            images_rgb = (images_rgb - images_rgb.min()) / (images_rgb.max() - images_rgb.min())
        
        # Convert to list of numpy arrays for ViT processor
        image_list = []
        for i in range(batch_size):
            img_np = images_rgb[i].permute(1, 2, 0).cpu().numpy()  # (28, 28, 3)
            image_list.append(img_np)
        
        # Process with ViT processor
        processed = self.vit_processor(image_list, return_tensors="pt")
        processed_images = processed['pixel_values'].to(images.device)
        
        return processed_images
    
    def encode_image(self, images):
        """Encode images using ViT-base-MNIST classifier as downstream task"""
        # Preprocess MNIST images for ViT
        processed_images = self.preprocess_images(images)
        
        # Use ViT classifier but extract features before final classification
        with torch.set_grad_enabled(self.training):
            # Get hidden states from ViT (before classification head)
            outputs = self.vit_classifier.vit(processed_images)
            # Use [CLS] token from last hidden state
            image_features = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)
        
        # Project to target embedding dimension
        image_features = self.image_projection(image_features)
        
        # L2 normalize
        image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(self, text_tokens):
        """Encode text using CLIP text encoder"""
        # Use CLIP's text encoding pipeline with proper dtype handling
        dtype = next(self.text_encoder.parameters()).dtype
        
        x = self.token_embedding(text_tokens).type(dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)
        
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection
        
        # L2 normalize
        text_features = F.normalize(x, p=2, dim=1)
        
        return text_features
    
    def forward(self, images, text_tokens):
        """
        Forward pass for contrastive learning
        
        Args:
            images: (batch_size, 1, 28, 28) MNIST images
            text_tokens: (batch_size, 77) CLIP-tokenized text
            
        Returns:
            image_features: (batch_size, output_dim)
            text_features: (batch_size, output_dim)
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
        
        # Ensure same dtype for matrix multiplication
        image_features = image_features.float()
        text_features = text_features.float()
        logit_scale = logit_scale.float()
        
        similarity = logit_scale * image_features @ text_features.T
        return similarity
    
    def classify_images(self, images):
        """
        Use ViT classifier for digit classification (downstream task)
        This demonstrates the proper use of the pre-trained model
        """
        processed_images = self.preprocess_images(images)
        
        with torch.no_grad():
            outputs = self.vit_classifier(processed_images)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
        
        return predicted_classes, predictions

def test_clip_downstream():
    """Test function for CLIP downstream task model"""
    print("Testing CLIP ViT Downstream Task...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize model
        model = CLIPViTDownstream(freeze_vit=False, device=device)
        
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
        import clip
        text_tokens = clip.tokenize(captions).to(device)
        
        print(f"Image shape: {images.shape}")
        print(f"Text tokens shape: {text_tokens.shape}")
        
        # Test downstream task (digit classification)
        print("\n🎯 Testing downstream task (digit classification):")
        predicted_classes, predictions = model.classify_images(images)
        print(f"Predicted classes: {predicted_classes}")
        print(f"Prediction confidence: {predictions.max(dim=-1)[0]}")
        
        # Test CLIP functionality
        print("\n🔗 Testing CLIP functionality:")
        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, text_tokens)
            
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Logit scale: {logit_scale.item():.4f}")
        
        # Test similarity
        similarity = model.compute_similarity(image_features, text_features)
        print(f"Similarity matrix shape: {similarity.shape}")
        print(f"Similarity values: {similarity.diag()}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n✅ CLIP Downstream Task test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_downstream()
