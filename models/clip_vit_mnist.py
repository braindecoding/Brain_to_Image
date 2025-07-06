#!/usr/bin/env python3

"""
CLIP-style ViT-B/32 Model for MNIST Digits

Purpose:
    CLIP-inspired model with ViT-B/32 image encoder and text encoder
    Trained with contrastive learning for image-text alignment

Architecture:
    - Image Encoder: ViT-B/32 (Vision Transformer)
    - Text Encoder: Transformer-based
    - Output: 512-dimensional aligned embeddings
    - Training: Contrastive learning with MNIST images + digit captions

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0

Usage:
    model = CLIPMNISTModel()
    image_features, text_features = model(images, text_tokens)
    similarity = model.compute_similarity(image_features, text_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for ViT
    Converts 28x28 MNIST images to patch embeddings
    """
    
    def __init__(self, img_size: int = 28, patch_size: int = 4, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 7x7 = 49 patches for 28x28 image
        
        # Patch embedding using convolution
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, 1, 28, 28)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size {H}x{W} doesn't match model size {self.img_size}x{self.img_size}"
        
        # Convert to patches: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        # Flatten patches: (batch_size, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for Transformer"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Head Attention and MLP"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT-B/32 style) for MNIST
    Adapted for 28x28 grayscale images
    """
    
    def __init__(self, img_size: int = 28, patch_size: int = 4, embed_dim: int = 768, 
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, output_dim: int = 512):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm and projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Use class token for final representation
        cls_output = x[:, 0]  # (B, embed_dim)
        
        # Project to output dimension
        output = self.head(cls_output)  # (B, output_dim)
        
        # L2 normalize for contrastive learning
        output = F.normalize(output, p=2, dim=1)
        
        return output


class TextEncoder(nn.Module):
    """
    Simple Text Encoder for MNIST digit captions
    Uses embedding + transformer for text understanding
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, max_length: int = 20,
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


class CLIPMNISTModel(nn.Module):
    """
    CLIP-style model for MNIST images and digit captions
    Combines ViT image encoder and transformer text encoder
    """

    def __init__(self,
                 # Image encoder params
                 img_size: int = 28,
                 patch_size: int = 4,
                 vision_embed_dim: int = 768,
                 vision_depth: int = 12,
                 vision_heads: int = 12,
                 # Text encoder params
                 vocab_size: int = 1000,
                 text_embed_dim: int = 512,
                 text_heads: int = 8,
                 text_layers: int = 6,
                 max_text_length: int = 20,
                 # Shared params
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 temperature: float = 0.07):
        super().__init__()

        self.temperature = temperature
        self.output_dim = output_dim

        # Image encoder (ViT)
        self.image_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_heads,
            output_dim=output_dim,
            dropout=dropout
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

    def encode_image(self, images):
        """Encode images to feature vectors"""
        return self.image_encoder(images)

    def encode_text(self, text_tokens, attention_mask=None):
        """Encode text to feature vectors"""
        return self.text_encoder(text_tokens, attention_mask)

    def forward(self, images, text_tokens, attention_mask=None):
        """
        Forward pass for contrastive learning

        Args:
            images: (batch_size, 1, 28, 28)
            text_tokens: (batch_size, seq_length)
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
        """
        Compute cosine similarity between image and text features

        Args:
            image_features: (batch_size, output_dim)
            text_features: (batch_size, output_dim)

        Returns:
            similarity: (batch_size, batch_size) similarity matrix
        """
        # Normalize features (should already be normalized)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * image_features @ text_features.T

        return similarity
