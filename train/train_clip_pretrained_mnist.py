#!/usr/bin/env python3

"""
Pre-trained CLIP Training for MNIST Digits

Purpose:
    Fine-tune pre-trained OpenAI CLIP ViT-B/32 for MNIST digits
    Much faster convergence and better performance than training from scratch

Pipeline:
    1. Load pre-trained CLIP ViT-B/32
    2. Adapt for MNIST: 28x28 → 224x224, grayscale → RGB
    3. Fine-tune with digit captions using contrastive learning
    4. Output: High-quality 512-dim aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v2.0 (Pre-trained)

Usage:
    python train/train_clip_pretrained_mnist.py
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_pretrained_mnist import CLIPPretrainedMNIST, CLIPTokenizer
from models.caption_generator import DigitCaptionGenerator
from utils.mnist_loader import MNISTLoader

class MNISTCLIPPretrainedDataset(Dataset):
    """
    Dataset for pre-trained CLIP fine-tuning with MNIST images and captions
    """
    
    def __init__(self, mnist_loader: MNISTLoader, caption_generator: DigitCaptionGenerator, 
                 split: str = 'train', samples_per_epoch: int = 10000):
        self.mnist_loader = mnist_loader
        self.caption_generator = caption_generator
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Get random digit (0-9)
        digit = np.random.randint(0, 10)
        
        # Get MNIST image for this digit
        image = self.mnist_loader.get_random_image_for_digit(digit, self.split)
        
        # Generate caption with variety
        if np.random.random() < 0.3:  # 30% chance for alternative caption
            template_idx = np.random.randint(0, 5)
            caption = self.caption_generator.generate_alternative_caption(digit, template_idx)
        else:
            caption = self.caption_generator.generate_caption(digit)
        
        return {
            'image': image,
            'caption': caption,
            'digit': digit
        }

def contrastive_loss(image_features, text_features, logit_scale, labels=None):
    """Compute contrastive loss (InfoNCE) for CLIP training"""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    similarity = logit_scale * image_features @ text_features.T
    
    # Create labels (diagonal should be positive pairs)
    if labels is None:
        labels = torch.arange(batch_size, device=image_features.device)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = nn.CrossEntropyLoss()(similarity, labels)
    loss_t2i = nn.CrossEntropyLoss()(similarity.T, labels)
    
    # Average both losses
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss

def train_epoch(model, dataloader, optimizer, tokenizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch['image'].to(device)
        captions = batch['caption']
        
        # Tokenize captions using CLIP tokenizer
        text_tokens = tokenizer.encode(captions).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # Mixed precision for efficiency
            image_features, text_features, logit_scale = model(images, text_tokens)
            loss = contrastive_loss(image_features, text_features, logit_scale)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Scale': f'{logit_scale.item():.2f}'
        })
    
    return total_loss / num_batches

def validate_model(model, dataloader, tokenizer, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_i2t = 0
    correct_t2i = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Tokenize captions
            text_tokens = tokenizer.encode(captions).to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = model(images, text_tokens)
                loss = contrastive_loss(image_features, text_features, logit_scale)
            
            total_loss += loss.item()
            
            # Compute accuracy
            similarity = logit_scale * image_features @ text_features.T
            
            # Image-to-text accuracy
            i2t_pred = torch.argmax(similarity, dim=1)
            correct_i2t += (i2t_pred == torch.arange(len(images), device=device)).sum().item()
            
            # Text-to-image accuracy
            t2i_pred = torch.argmax(similarity.T, dim=1)
            correct_t2i += (t2i_pred == torch.arange(len(images), device=device)).sum().item()
            
            total_samples += len(images)
    
    avg_loss = total_loss / len(dataloader)
    i2t_acc = correct_i2t / total_samples
    t2i_acc = correct_t2i / total_samples
    
    return avg_loss, i2t_acc, t2i_acc

def main():
    """Main training function"""
    print("🚀 Starting Pre-trained CLIP Fine-tuning for MNIST")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    mnist_loader = MNISTLoader()
    caption_generator = DigitCaptionGenerator()
    tokenizer = CLIPTokenizer()
    
    # Test tokenization
    print("\n📝 Testing CLIP tokenization...")
    test_captions = ["A handwritten digit zero", "A handwritten digit five"]
    test_tokens = tokenizer.encode(test_captions)
    print(f"Caption: '{test_captions[0]}'")
    print(f"Tokens shape: {test_tokens.shape}")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = MNISTCLIPPretrainedDataset(mnist_loader, caption_generator, 'train', 20000)
    val_dataset = MNISTCLIPPretrainedDataset(mnist_loader, caption_generator, 'test', 2000)
    
    # Create dataloaders
    batch_size = 64  # Smaller batch for pre-trained model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️ Creating Pre-trained CLIP model...")
    model = CLIPPretrainedMNIST(
        device=device,
        freeze_backbone=False,  # Fine-tune for better adaptation
        fine_tune_layers=3      # Fine-tune last 3 layers
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup training - LOWER learning rate for pre-trained model
    num_epochs = 30  # Fewer epochs needed for pre-trained
    learning_rate = 1e-5  # Much lower LR for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n🎯 Pre-trained CLIP Training setup:")
    print(f"  Epochs: {num_epochs} (fewer needed for pre-trained)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (low for fine-tuning)")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Fine-tuning: Last 3 layers")
    print(f"  Expected: MUCH better accuracy than from-scratch")
    
    # Training loop
    print(f"\n🚀 Starting pre-trained CLIP fine-tuning...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, tokenizer, device, epoch)
        
        # Validate
        val_loss, i2t_acc, t2i_acc = validate_model(model, val_loader, tokenizer, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Image→Text Acc: {i2t_acc:.4f}")
        print(f"Text→Image Acc: {t2i_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/clip_pretrained_mnist_best.pth')
            print("✅ Best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/clip_pretrained_mnist_final.pth')
    print(f"\n🎉 Pre-trained CLIP fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
