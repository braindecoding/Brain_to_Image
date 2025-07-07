#!/usr/bin/env python3

"""
CLIP Embedding Decoder for Image Reconstruction

Purpose:
    Decode rich 512-dimensional CLIP embeddings back to MNIST images
    Uses deep neural network to reconstruct images from semantic embeddings

Architecture:
    - Input: 512-dim CLIP embeddings (from Pure CLIP Frozen)
    - Hidden: Multi-layer MLP with residual connections
    - Output: 28x28 MNIST images (1 channel grayscale)
    - Training: MSE loss + perceptual loss for high-quality reconstruction

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0 (CLIP Decoder)

Usage:
    decoder = CLIPDecoder()
    reconstructed_images = decoder(clip_embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CLIPDecoder(nn.Module):
    """
    Decoder to reconstruct MNIST images from CLIP embeddings
    """
    
    def __init__(self, 
                 input_dim: int = 512,           # CLIP embedding dimension
                 hidden_dims: list = [1024, 2048, 1024, 512],  # Hidden layer dimensions
                 output_size: tuple = (28, 28),  # MNIST image size
                 num_residual_blocks: int = 3,   # Number of residual blocks
                 dropout: float = 0.1,           # Dropout rate
                 activation: str = 'relu'):      # Activation function
        super().__init__()
        
        self.input_dim = input_dim
        self.output_size = output_size
        self.output_dim = output_size[0] * output_size[1]  # 784 for 28x28
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers with residual connections
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Residual blocks for better feature learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout) for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.output_dim),
            nn.Sigmoid()  # Output in [0, 1] range for images
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ CLIP Decoder initialized")
        print(f"   Input: {input_dim}-dim CLIP embeddings")
        print(f"   Output: {output_size} images")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Residual blocks: {num_residual_blocks}")
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, embeddings):
        """
        Decode CLIP embeddings to images
        
        Args:
            embeddings: (batch_size, input_dim) CLIP embeddings
            
        Returns:
            images: (batch_size, 1, height, width) reconstructed images
        """
        batch_size = embeddings.shape[0]
        
        # Input projection
        x = self.input_projection(embeddings)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Reshape to image format
        images = x.view(batch_size, 1, self.output_size[0], self.output_size[1])
        
        return images

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using a simple CNN feature extractor
    Helps generate more realistic images
    """
    
    def __init__(self):
        super().__init__()
        
        # Simple CNN for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # Freeze feature extractor (optional)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred_images, target_images):
        """Compute perceptual loss between predicted and target images"""
        pred_features = self.feature_extractor(pred_images)
        target_features = self.feature_extractor(target_images)
        
        loss = F.mse_loss(pred_features, target_features)
        return loss

class CLIPDecoderLoss(nn.Module):
    """
    Combined loss for CLIP decoder training
    """
    
    def __init__(self, mse_weight: float = 1.0, perceptual_weight: float = 0.1):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(self, pred_images, target_images):
        """Compute combined loss"""
        # MSE loss for pixel-level reconstruction
        mse = self.mse_loss(pred_images, target_images)
        
        # Perceptual loss for realistic appearance
        perceptual = self.perceptual_loss(pred_images, target_images)
        
        # Combined loss
        total_loss = self.mse_weight * mse + self.perceptual_weight * perceptual
        
        return total_loss, mse, perceptual

def test_clip_decoder():
    """Test function for CLIP decoder"""
    print("Testing CLIP Decoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize decoder
        decoder = CLIPDecoder().to(device)
        
        # Test data
        batch_size = 8
        clip_embeddings = torch.randn(batch_size, 512).to(device)  # Rich CLIP embeddings
        target_images = torch.randn(batch_size, 1, 28, 28).to(device)  # Target MNIST images
        
        print(f"CLIP embeddings shape: {clip_embeddings.shape}")
        print(f"Target images shape: {target_images.shape}")
        
        # Forward pass
        with torch.no_grad():
            reconstructed_images = decoder(clip_embeddings)
            
        print(f"Reconstructed images shape: {reconstructed_images.shape}")
        print(f"Output range: [{reconstructed_images.min():.4f}, {reconstructed_images.max():.4f}]")
        
        # Test loss function
        loss_fn = CLIPDecoderLoss().to(device)

        # Test with gradient computation
        clip_embeddings.requires_grad_(True)
        reconstructed_images = decoder(clip_embeddings)
        total_loss, mse_loss, perceptual_loss = loss_fn(reconstructed_images, target_images)

        print(f"Total loss: {total_loss.item():.4f}")
        print(f"MSE loss: {mse_loss.item():.4f}")
        print(f"Perceptual loss: {perceptual_loss.item():.4f}")

        # Count parameters
        total_params = sum(p.numel() for p in decoder.parameters())
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Test gradient flow
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        print(f"Gradient norm: {grad_norm:.4f}")
        
        print("\n✅ CLIP Decoder test completed!")
        
        # Test with different embedding dimensions
        print("\n🧪 Testing different embedding dimensions:")
        for embed_dim in [256, 512, 768, 1024]:
            test_decoder = CLIPDecoder(input_dim=embed_dim, hidden_dims=[512, 1024, 512]).to(device)
            test_embeddings = torch.randn(4, embed_dim).to(device)
            
            with torch.no_grad():
                test_output = test_decoder(test_embeddings)
            
            print(f"  {embed_dim}-dim → {test_output.shape} ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_decoder()
