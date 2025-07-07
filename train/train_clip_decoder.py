#!/usr/bin/env python3

"""
CLIP Decoder Training Script

Purpose:
    Train CLIP decoder to reconstruct MNIST images from rich CLIP embeddings
    Uses Pure CLIP (Frozen) to generate high-quality embeddings (similarity 3.7+)
    Then trains decoder to map 512-dim embeddings → 28x28 images

Pipeline:
    1. Load Pure CLIP (Frozen) - no training needed
    2. Generate CLIP embeddings for MNIST dataset
    3. Train decoder: embeddings → images
    4. Save trained decoder for end-to-end pipeline

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0 (Decoder Training)

Usage:
    python train/train_clip_decoder.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_vit32_simple import CLIPViTSimple
from models.clip_decoder import CLIPDecoder, CLIPDecoderLoss
from models.caption_generator import DigitCaptionGenerator
from utils.mnist_loader import MNISTLoader

class CLIPDecoderDataset(Dataset):
    """
    Dataset for CLIP decoder training
    Generates CLIP embeddings and corresponding MNIST images
    """
    
    def __init__(self, mnist_loader: MNISTLoader, clip_model: CLIPViTSimple, 
                 caption_generator: DigitCaptionGenerator, split: str = 'train', 
                 samples_per_epoch: int = 10000, device: str = 'cuda'):
        self.mnist_loader = mnist_loader
        self.clip_model = clip_model
        self.caption_generator = caption_generator
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.device = device
        
        # Pre-generate embeddings for efficiency (optional)
        self.precompute_embeddings = False
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Get random digit (0-9)
        digit = np.random.randint(0, 10)
        
        # Get MNIST image for this digit
        image = self.mnist_loader.get_random_image_for_digit(digit, self.split)
        
        # Generate caption
        caption = self.caption_generator.generate_caption(digit)
        
        # Generate CLIP embedding using Pure CLIP (Frozen)
        with torch.no_grad():
            # Tokenize caption
            text_tokens = clip.tokenize([caption]).to(self.device)
            image_batch = image.unsqueeze(0).to(self.device)
            
            # Get CLIP embeddings
            image_features, text_features, _ = self.clip_model(image_batch, text_tokens)
            
            # Use image features as rich embedding (similarity 3.7+)
            clip_embedding = image_features.squeeze(0).cpu()
        
        return {
            'clip_embedding': clip_embedding,  # (512,) rich semantic features
            'target_image': image.squeeze(0),  # (28, 28) target for reconstruction
            'digit': digit,
            'caption': caption
        }

def train_epoch(decoder, dataloader, optimizer, loss_fn, device, epoch):
    """Train decoder for one epoch"""
    decoder.train()
    total_loss = 0
    total_mse = 0
    total_perceptual = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        clip_embeddings = batch['clip_embedding'].to(device)  # (batch_size, 512)
        target_images = batch['target_image'].unsqueeze(1).to(device)  # (batch_size, 1, 28, 28)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed_images = decoder(clip_embeddings)
        
        # Compute loss
        total_loss_batch, mse_loss, perceptual_loss = loss_fn(reconstructed_images, target_images)
        
        # Backward pass
        total_loss_batch.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        total_mse += mse_loss.item()
        total_perceptual += perceptual_loss.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_mse = total_mse / (batch_idx + 1)
        avg_perceptual = total_perceptual / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'MSE': f'{avg_mse:.4f}',
            'Perceptual': f'{avg_perceptual:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return avg_loss, avg_mse, avg_perceptual

def validate_decoder(decoder, dataloader, loss_fn, device):
    """Validate decoder"""
    decoder.eval()
    total_loss = 0
    total_mse = 0
    total_perceptual = 0
    
    with torch.no_grad():
        for batch in dataloader:
            clip_embeddings = batch['clip_embedding'].to(device)
            target_images = batch['target_image'].unsqueeze(1).to(device)
            
            # Forward pass
            reconstructed_images = decoder(clip_embeddings)
            
            # Compute loss
            total_loss_batch, mse_loss, perceptual_loss = loss_fn(reconstructed_images, target_images)
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_perceptual += perceptual_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_perceptual = total_perceptual / len(dataloader)
    
    return avg_loss, avg_mse, avg_perceptual

def save_sample_reconstructions(decoder, dataloader, device, epoch, save_dir='results'):
    """Save sample reconstructions for visual inspection"""
    os.makedirs(save_dir, exist_ok=True)
    
    decoder.eval()
    with torch.no_grad():
        # Get one batch
        batch = next(iter(dataloader))
        clip_embeddings = batch['clip_embedding'][:8].to(device)  # First 8 samples
        target_images = batch['target_image'][:8]
        captions = batch['caption'][:8]
        
        # Reconstruct
        reconstructed_images = decoder(clip_embeddings).cpu()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # Original image
            axes[0, i].imshow(target_images[i], cmap='gray')
            axes[0, i].set_title(f'Original\n{captions[i][:15]}...', fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed_images[i, 0], cmap='gray')
            axes[1, i].set_title('Reconstructed', fontsize=8)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    print("🚀 Starting CLIP Decoder Training")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    mnist_loader = MNISTLoader()
    caption_generator = DigitCaptionGenerator()
    
    # Load Pure CLIP (Frozen) for embedding generation
    print("\n🔥 Loading Pure CLIP (Frozen) for rich embeddings...")
    clip_model = CLIPViTSimple(
        device=device,
        freeze_backbone=True,      # FROZEN for stability
        adaptation_layers=False    # Pure CLIP
    )
    clip_model.eval()  # Always in eval mode
    
    print(f"✅ Pure CLIP loaded with similarity 3.7+ capability")
    
    # Create datasets
    print("\n📊 Creating decoder training datasets...")
    train_dataset = CLIPDecoderDataset(mnist_loader, clip_model, caption_generator, 'train', 20000, device)
    val_dataset = CLIPDecoderDataset(mnist_loader, clip_model, caption_generator, 'test', 2000, device)
    
    # Create dataloaders - no multiprocessing for CUDA compatibility
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create decoder
    print("\n🏗️ Creating CLIP Decoder...")
    decoder = CLIPDecoder(
        input_dim=512,                           # CLIP embedding dimension
        hidden_dims=[1024, 2048, 1024, 512],   # Rich hidden layers
        output_size=(28, 28),                   # MNIST size
        num_residual_blocks=3,                  # Deep architecture
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {total_params:,}")
    
    # Setup training
    num_epochs = 30
    learning_rate = 1e-4  # Standard for decoder training
    optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = CLIPDecoderLoss(mse_weight=1.0, perceptual_weight=0.1).to(device)
    
    print(f"\n🎯 CLIP Decoder Training setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: AdamW")
    print(f"  Loss: MSE + Perceptual (0.1)")
    print(f"  Input: Rich CLIP embeddings (similarity 3.7+)")
    print(f"  Output: 28x28 MNIST reconstructions")
    print(f"  Expected: High-quality image reconstruction")
    
    # Training loop
    print(f"\n🚀 Starting CLIP decoder training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_mse, train_perceptual = train_epoch(
            decoder, train_loader, optimizer, loss_fn, device, epoch
        )
        
        # Validate
        val_loss, val_mse, val_perceptual = validate_decoder(
            decoder, val_loader, loss_fn, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, Perceptual: {train_perceptual:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, Perceptual: {val_perceptual:.4f}")
        
        # Save sample reconstructions every 5 epochs
        if epoch % 5 == 0:
            save_sample_reconstructions(decoder, val_loader, device, epoch)
            print(f"📸 Sample reconstructions saved for epoch {epoch}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), 'models/clip_decoder_best.pth')
            print("✅ Best decoder model saved!")
    
    # Save final model
    torch.save(decoder.state_dict(), 'models/clip_decoder_final.pth')
    print(f"\n🎉 CLIP decoder training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: clip_decoder_best.pth, clip_decoder_final.pth")

if __name__ == "__main__":
    main()
