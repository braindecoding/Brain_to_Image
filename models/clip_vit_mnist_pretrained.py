#!/usr/bin/env python3

"""
CLIP with Pre-trained ViT-base-MNIST Transfer Learning

Purpose:
    Use pre-trained ViT-base-MNIST as image encoder in CLIP architecture
    Much better starting point since it's already trained on MNIST digits

Architecture:
    - Image Encoder: Pre-trained ViT-base-MNIST from HuggingFace
    - Text Encoder: Custom transformer for digit captions
    - Training: Contrastive learning for image-text alignment
    - Output: 512-dimensional aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v3.0 (ViT-MNIST Transfer Learning)

Usage:
    model = CLIPViTMNIST()
    image_features, text_features = model(images, text_tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from typing import Tuple, List
import numpy as np

class TextEncoder(nn.Module):
    """
    Simple Text Encoder for MNIST digit captions
    Uses embedding + transformer for text understanding
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 4, max_length: int = 20,
                 output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, text_tokens, attention_mask=None):
        # text_tokens: (batch_size, seq_length)
        batch_size, seq_length = text_tokens.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, device=text_tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(text_tokens)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (text_tokens != 0).float()  # Assume 0 is padding token
        
        # Convert attention mask for transformer (True = ignore)
        attention_mask = (attention_mask == 0)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Use the first token (like [CLS]) for text representation
        x = self.ln_final(x[:, 0])  # (batch_size, embed_dim)
        
        # Project to output dimension
        text_features = self.text_projection(x)  # (batch_size, output_dim)
        
        # L2 normalize for contrastive learning
        text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features

class CLIPViTMNIST(nn.Module):
    """
    CLIP model with pre-trained ViT-base-MNIST as image encoder
    Uses HuggingFace pre-trained ViT specifically trained on MNIST
    """
    
    def __init__(self, 
                 # Pre-trained ViT model
                 model_name: str = "farleyknight-org-username/vit-base-mnist",
                 # Text encoder params
                 vocab_size: int = 1000,
                 text_embed_dim: int = 512,
                 text_heads: int = 8,
                 text_layers: int = 4,
                 max_text_length: int = 20,
                 # Shared params
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 temperature: float = 0.07,
                 freeze_vit: bool = False):
        super().__init__()
        
        self.output_dim = output_dim
        self.model_name = model_name
        
        # Load pre-trained ViT-base-MNIST
        print(f"📥 Loading pre-trained ViT-base-MNIST: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit_model = ViTForImageClassification.from_pretrained(model_name)
        
        # Get ViT hidden size
        self.vit_hidden_size = self.vit_model.config.hidden_size  # Usually 768
        
        # Freeze ViT if requested
        if freeze_vit:
            print("🔒 Freezing ViT-base-MNIST weights...")
            for param in self.vit_model.parameters():
                param.requires_grad = False
        else:
            print("🔧 Fine-tuning ViT-base-MNIST...")
        
        # Image projection layer (ViT hidden_size -> output_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(self.vit_hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            max_length=max_text_length,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        print(f"✅ CLIP with ViT-base-MNIST loaded")
        print(f"   ViT hidden size: {self.vit_hidden_size}")
        print(f"   Output dimension: {output_dim}")
        print(f"   Frozen ViT: {freeze_vit}")
    
    def preprocess_images(self, images):
        """
        Preprocess MNIST images for ViT-base-MNIST
        Convert from tensor format to PIL-like format expected by processor
        """
        # images: (batch_size, 1, 28, 28)
        batch_size = images.shape[0]
        
        # Convert to 3-channel if needed (some ViT models expect RGB)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # (batch_size, 3, 28, 28)
        
        # Normalize to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # ViT processor expects values in [0, 1] range
        # Convert to list of numpy arrays for processor
        image_list = []
        for i in range(batch_size):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()  # (28, 28, 3)
            image_list.append(img_np)
        
        # Process with ViT processor
        processed = self.processor(image_list, return_tensors="pt")
        
        # Move to same device as input
        processed_images = processed['pixel_values'].to(images.device)
        
        return processed_images
    
    def encode_image(self, images):
        """Encode images using pre-trained ViT-base-MNIST"""
        # Preprocess images
        processed_images = self.preprocess_images(images)
        
        # Get ViT features (use hidden states, not classification logits)
        with torch.set_grad_enabled(self.training):
            outputs = self.vit_model(processed_images, output_hidden_states=True)
            # Use [CLS] token from last hidden state
            image_features = outputs.hidden_states[-1][:, 0]  # (batch_size, hidden_size)
        
        # Project to output dimension
        image_features = self.image_projection(image_features)
        
        # L2 normalize
        image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(self, text_tokens, attention_mask=None):
        """Encode text using custom text encoder"""
        return self.text_encoder(text_tokens, attention_mask)
    
    def forward(self, images, text_tokens, attention_mask=None):
        """
        Forward pass for contrastive learning
        
        Args:
            images: (batch_size, 1, 28, 28) MNIST images
            text_tokens: (batch_size, seq_length) tokenized text
            attention_mask: (batch_size, seq_length) optional
            
        Returns:
            image_features: (batch_size, output_dim)
            text_features: (batch_size, output_dim)
            logit_scale: scalar
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens, attention_mask)
        
        return image_features, text_features, self.logit_scale.exp()
    
    def compute_similarity(self, image_features, text_features):
        """Compute cosine similarity between image and text features"""
        # Features should already be normalized
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * image_features @ text_features.T
        return similarity

def test_clip_vit_mnist():
    """Test function for CLIP with ViT-base-MNIST"""
    print("Testing CLIPViTMNIST...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize model
        model = CLIPViTMNIST(freeze_vit=False).to(device)
        
        # Test data
        batch_size = 4
        images = torch.randn(batch_size, 1, 28, 28).to(device)  # MNIST format
        
        # Simple tokenizer for testing
        text_tokens = torch.randint(1, 100, (batch_size, 10)).to(device)
        
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
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n✅ CLIP with ViT-base-MNIST test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_vit_mnist()
