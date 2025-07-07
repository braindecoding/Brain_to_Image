#!/usr/bin/env python3

"""
CLIP Downstream Task Training for MNIST

Purpose:
    Train CLIP model using ViTForImageClassification as proper downstream task
    Combines digit classification loss + contrastive learning for optimal performance

Pipeline:
    1. Load ViT-base-MNIST classifier (proper downstream task)
    2. Add CLIP text encoder for digit captions
    3. Multi-task training: classification loss + contrastive loss
    4. Output: High-quality 512-dim aligned embeddings + digit classification

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v4.0 (Downstream Task Training)

Usage:
    python train/train_clip_downstream.py
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

from models.clip_vit_downstream import CLIPViTDownstream
from models.caption_generator import DigitCaptionGenerator
from utils.mnist_loader import MNISTLoader

class MNISTCLIPDownstreamDataset(Dataset):
    """
    Dataset for CLIP downstream task training with MNIST images and captions
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
            'digit': digit,
            'label': torch.tensor(digit, dtype=torch.long)  # For classification loss
        }

def contrastive_loss(image_features, text_features, logit_scale, labels=None):
    """Compute contrastive loss (InfoNCE) for CLIP training"""
    batch_size = image_features.shape[0]
    
    # Ensure same dtype for all tensors
    image_features = image_features.float()
    text_features = text_features.float()
    logit_scale = logit_scale.float()
    
    # Clamp logit_scale for stability - MORE CONSERVATIVE
    logit_scale = torch.clamp(logit_scale, max=5.0)  # Lower limit to prevent NaN
    
    # Compute similarity matrix
    similarity = logit_scale * image_features @ text_features.T

    # Check for NaN/Inf in similarity matrix
    if torch.isnan(similarity).any() or torch.isinf(similarity).any():
        print(f"⚠️ NaN/Inf detected in similarity matrix! Returning zero loss.")
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # Create labels (diagonal should be positive pairs)
    if labels is None:
        labels = torch.arange(batch_size, device=image_features.device)

    # Compute cross-entropy loss for both directions
    loss_i2t = nn.CrossEntropyLoss()(similarity, labels)
    loss_t2i = nn.CrossEntropyLoss()(similarity.T, labels)

    # Check for NaN in losses
    if torch.isnan(loss_i2t) or torch.isnan(loss_t2i):
        print(f"⚠️ NaN detected in cross-entropy loss! Returning zero loss.")
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # Average both losses
    loss = (loss_i2t + loss_t2i) / 2

    return loss

def classification_loss(model, images, labels):
    """Compute classification loss using ViT downstream task"""
    # Use the classification head of ViT
    processed_images = model.preprocess_images(images)
    outputs = model.vit_classifier(processed_images)
    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
    return loss, outputs.logits

def train_epoch(model, dataloader, optimizer, device, epoch, alpha=0.5):
    """Train for one epoch with multi-task learning"""
    model.train()
    total_loss = 0
    total_clip_loss = 0
    total_class_loss = 0
    correct_classifications = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch['image'].to(device)
        captions = batch['caption']
        labels = batch['label'].to(device)
        
        # Tokenize captions using CLIP tokenizer
        text_tokens = clip.tokenize(captions).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # CLIP contrastive learning
        image_features, text_features, logit_scale = model(images, text_tokens)
        clip_loss = contrastive_loss(image_features, text_features, logit_scale)
        
        # Classification loss (downstream task)
        class_loss, class_logits = classification_loss(model, images, labels)
        
        # Combined loss
        total_loss_batch = alpha * clip_loss + (1 - alpha) * class_loss
        
        # Backward pass
        total_loss_batch.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        total_clip_loss += clip_loss.item()
        total_class_loss += class_loss.item()
        
        # Classification accuracy
        predicted = torch.argmax(class_logits, dim=1)
        correct_classifications += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_clip_loss = total_clip_loss / (batch_idx + 1)
        avg_class_loss = total_class_loss / (batch_idx + 1)
        class_acc = correct_classifications / total_samples
        
        progress_bar.set_postfix({
            'Total': f'{avg_loss:.4f}',
            'CLIP': f'{avg_clip_loss:.4f}',
            'Class': f'{avg_class_loss:.4f}',
            'Acc': f'{class_acc:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return avg_loss, avg_clip_loss, avg_class_loss, class_acc

def validate_model(model, dataloader, device, alpha=0.5):
    """Validate model with both CLIP and classification metrics"""
    model.eval()
    total_loss = 0
    total_clip_loss = 0
    total_class_loss = 0
    correct_i2t = 0
    correct_t2i = 0
    correct_classifications = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            captions = batch['caption']
            labels = batch['label'].to(device)
            
            # Tokenize captions
            text_tokens = clip.tokenize(captions).to(device)
            
            # Forward pass
            image_features, text_features, logit_scale = model(images, text_tokens)
            clip_loss = contrastive_loss(image_features, text_features, logit_scale)
            
            # Classification loss
            class_loss, class_logits = classification_loss(model, images, labels)
            
            # Combined loss
            total_loss_batch = alpha * clip_loss + (1 - alpha) * class_loss
            
            total_loss += total_loss_batch.item()
            total_clip_loss += clip_loss.item()
            total_class_loss += class_loss.item()
            
            # CLIP accuracy
            similarity = logit_scale * image_features @ text_features.T
            
            # Image-to-text accuracy
            i2t_pred = torch.argmax(similarity, dim=1)
            correct_i2t += (i2t_pred == torch.arange(len(images), device=device)).sum().item()
            
            # Text-to-image accuracy
            t2i_pred = torch.argmax(similarity.T, dim=1)
            correct_t2i += (t2i_pred == torch.arange(len(images), device=device)).sum().item()
            
            # Classification accuracy
            predicted = torch.argmax(class_logits, dim=1)
            correct_classifications += (predicted == labels).sum().item()
            
            total_samples += len(images)
    
    avg_loss = total_loss / len(dataloader)
    avg_clip_loss = total_clip_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    i2t_acc = correct_i2t / total_samples
    t2i_acc = correct_t2i / total_samples
    class_acc = correct_classifications / total_samples
    
    return avg_loss, avg_clip_loss, avg_class_loss, i2t_acc, t2i_acc, class_acc

def main():
    """Main training function"""
    print("🚀 Starting CLIP Downstream Task Training")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    mnist_loader = MNISTLoader()
    caption_generator = DigitCaptionGenerator()
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = MNISTCLIPDownstreamDataset(mnist_loader, caption_generator, 'train', 30000)
    val_dataset = MNISTCLIPDownstreamDataset(mnist_loader, caption_generator, 'test', 3000)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️ Creating CLIP Downstream Task model...")
    model = CLIPViTDownstream(
        output_dim=512,
        temperature=0.07,
        freeze_vit=True,   # FREEZE ViT for maximum stability
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training - ULTRA CONSERVATIVE for stability
    num_epochs = 25
    learning_rate = 2e-6  # ULTRA LOW LR to prevent NaN
    alpha = 0.3  # Lower weight for CLIP (more stable classification focus)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)  # Minimal weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n🎯 CLIP Downstream Task Training setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Alpha (CLIP vs Class): {alpha:.1f} vs {1-alpha:.1f}")
    print(f"  Optimizer: AdamW (ultra conservative)")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Architecture: ViTForImageClassification + CLIP text")
    print(f"  Multi-task: Contrastive + Classification")
    print(f"  ViT: FROZEN for maximum stability")
    print(f"  Expected: 20-40% CLIP accuracy + 85%+ classification (ultra conservative)")
    
    # Training loop
    print(f"\n🚀 Starting CLIP downstream task training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_clip_loss, train_class_loss, train_class_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, alpha
        )
        
        # Validate
        val_loss, val_clip_loss, val_class_loss, i2t_acc, t2i_acc, val_class_acc = validate_model(
            model, val_loader, device, alpha
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train - Total: {train_loss:.4f}, CLIP: {train_clip_loss:.4f}, Class: {train_class_loss:.4f}, Acc: {train_class_acc:.4f}")
        print(f"Val   - Total: {val_loss:.4f}, CLIP: {val_clip_loss:.4f}, Class: {val_class_loss:.4f}, Acc: {val_class_acc:.4f}")
        print(f"CLIP  - Image→Text: {i2t_acc:.4f}, Text→Image: {t2i_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/clip_downstream_best.pth')
            print("✅ Best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/clip_downstream_final.pth')
    print(f"\n🎉 CLIP downstream task training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
