#!/usr/bin/env python3

"""
CLIP Training with Pre-trained ViT-base-MNIST

Purpose:
    Train CLIP model using pre-trained ViT-base-MNIST as image encoder
    Should achieve much better performance since ViT is already trained on MNIST

Pipeline:
    1. Load pre-trained ViT-base-MNIST (already knows MNIST digits)
    2. Add text encoder for digit captions
    3. Train with contrastive learning for image-text alignment
    4. Output: High-quality 512-dim aligned embeddings

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v3.0 (ViT-MNIST Transfer Learning)

Usage:
    python train/train_clip_vit_mnist.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_vit_mnist_pretrained import CLIPViTMNIST
from models.caption_generator import DigitCaptionGenerator
from utils.mnist_loader import MNISTLoader

class SimpleTokenizer:
    """Simple tokenizer for digit captions"""
    
    def __init__(self):
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'a': 3, 'handwritten': 4, 'digit': 5,
            'zero': 6, 'one': 7, 'two': 8, 'three': 9, 'four': 10,
            'five': 11, 'six': 12, 'seven': 13, 'eight': 14, 'nine': 15,
            'the': 16, 'number': 17, 'written': 18
        }
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str, max_length: int = 20):
        """Encode text to token IDs"""
        tokens = ['<START>'] + text.lower().split() + ['<END>']
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)

class MNISTCLIPViTDataset(Dataset):
    """Dataset for CLIP training with ViT-base-MNIST"""
    
    def __init__(self, mnist_loader: MNISTLoader, caption_generator: DigitCaptionGenerator, 
                 tokenizer: SimpleTokenizer, split: str = 'train', samples_per_epoch: int = 10000):
        self.mnist_loader = mnist_loader
        self.caption_generator = caption_generator
        self.tokenizer = tokenizer
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
        
        # Tokenize caption
        text_tokens = self.tokenizer.encode(caption)
        
        return {
            'image': image,
            'text_tokens': text_tokens,
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

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch['image'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
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

def validate_model(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct_i2t = 0
    correct_t2i = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            
            # Forward pass
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
    print("🚀 Starting CLIP Training with Pre-trained ViT-base-MNIST")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    mnist_loader = MNISTLoader()
    caption_generator = DigitCaptionGenerator()
    tokenizer = SimpleTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = MNISTCLIPViTDataset(mnist_loader, caption_generator, tokenizer, 'train', 30000)
    val_dataset = MNISTCLIPViTDataset(mnist_loader, caption_generator, tokenizer, 'test', 3000)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🏗️ Creating CLIP with ViT-base-MNIST...")
    model = CLIPViTMNIST(
        vocab_size=tokenizer.vocab_size,
        output_dim=512,
        temperature=0.07,
        freeze_vit=False  # Fine-tune ViT for better adaptation
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    num_epochs = 30
    learning_rate = 1e-4  # Standard learning rate for ViT fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n🎯 ViT-base-MNIST CLIP Training setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  ViT: Pre-trained on MNIST (fine-tuned)")
    print(f"  Expected: MUCH better accuracy (ViT already knows MNIST!)")
    
    # Training loop
    print(f"\n🚀 Starting ViT-base-MNIST CLIP training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, i2t_acc, t2i_acc = validate_model(model, val_loader, device)
        
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
            torch.save(model.state_dict(), 'models/clip_vit_mnist_best.pth')
            print("✅ Best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/clip_vit_mnist_final.pth')
    print(f"\n🎉 ViT-base-MNIST CLIP training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
