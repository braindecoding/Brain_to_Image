#!/usr/bin/env python3

"""
CLIP-style Training for MNIST Images and Digit Captions

Purpose:
    Train CLIP-inspired model with contrastive learning
    Aligns MNIST images with descriptive digit captions
    
Pipeline:
    1. EEG signals → PyTorch classifier → digit predictions
    2. Digit predictions → template captions ("A handwritten digit {number}")
    3. EEG labels → MNIST images → ViT-B/32 CLIP training
    4. Output: 512-dim aligned embeddings for cross-modal retrieval

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0

Usage:
    python train/train_clip_mnist.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_vit_mnist import CLIPMNISTModel
from models.caption_generator import DigitCaptionGenerator
from utils.mnist_loader import MNISTLoader
from utils.eeg_inference import EEGInferencer

class SimpleTokenizer:
    """
    Simple tokenizer for digit captions
    Maps words to token IDs for text encoder
    """
    
    def __init__(self):
        # Build vocabulary from digit captions
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'a': 3, 'handwritten': 4, 'digit': 5,
            'zero': 6, 'one': 7, 'two': 8, 'three': 9, 'four': 10,
            'five': 11, 'six': 12, 'seven': 13, 'eight': 14, 'nine': 15,
            'the': 16, 'number': 17, 'written': 18
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str, max_length: int = 20):
        """Encode text to token IDs"""
        tokens = ['<START>'] + text.lower().split() + ['<END>']
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Skip unknown tokens for simplicity
                continue
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id == self.vocab['<PAD>']:
                break
            tokens.append(self.id_to_token.get(token_id.item(), '<UNK>'))
        return ' '.join(tokens)

class MNISTCLIPDataset(Dataset):
    """
    Dataset for CLIP training with MNIST images and captions
    """
    
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

        # Generate caption with VARIETY (use alternative templates sometimes)
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
            'digit': digit,
            'caption': caption
        }

def contrastive_loss(image_features, text_features, logit_scale, labels=None):
    """
    Compute contrastive loss (InfoNCE) for CLIP training
    
    Args:
        image_features: (batch_size, embed_dim)
        text_features: (batch_size, embed_dim)
        logit_scale: temperature parameter
        labels: ground truth labels (optional)
    
    Returns:
        loss: contrastive loss
    """
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
        
        # Compute loss
        loss = contrastive_loss(image_features, text_features, logit_scale)
        
        # Backward pass
        loss.backward()
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
            
            # Compute loss
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
    print("🚀 Starting CLIP-style MNIST Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize components
    print("\n📦 Initializing components...")
    mnist_loader = MNISTLoader()
    caption_generator = DigitCaptionGenerator()
    tokenizer = SimpleTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test caption generation
    print("\n📝 Testing caption generation...")
    for digit in range(10):
        caption = caption_generator.generate_caption(digit)
        tokens = tokenizer.encode(caption)
        decoded = tokenizer.decode(tokens)
        print(f"  {digit}: {caption} -> {decoded}")
    
    # Create datasets with LARGER size for better learning
    print("\n📊 Creating LARGER datasets...")
    train_dataset = MNISTCLIPDataset(mnist_loader, caption_generator, tokenizer, 'train', 50000)  # 5x larger
    val_dataset = MNISTCLIPDataset(mnist_loader, caption_generator, tokenizer, 'test', 5000)  # 5x larger
    
    # Create dataloaders with OPTIMIZED batch size
    batch_size = 128  # LARGER batch size for better GPU utilization and stable gradients
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model with IMPROVED architecture
    print("\n🏗️ Creating ENHANCED CLIP model...")
    model = CLIPMNISTModel(
        # Vision encoder improvements
        vision_embed_dim=768,  # Keep standard ViT-B size
        vision_depth=8,        # REDUCED depth for faster training
        vision_heads=12,       # Keep standard
        # Text encoder improvements
        vocab_size=tokenizer.vocab_size,
        text_embed_dim=512,    # Standard size
        text_heads=8,          # Standard
        text_layers=4,         # REDUCED layers for faster training
        # Output improvements
        output_dim=512,
        temperature=0.05       # LOWER temperature for sharper distributions
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training with IMPROVED parameters
    num_epochs = 100  # DOUBLED epochs for better convergence
    learning_rate = 2e-4  # HIGHER learning rate for faster learning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)  # REDUCED weight decay

    # IMPROVED scheduler with warmup
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    
    print(f"\n🎯 ENHANCED Training setup:")
    print(f"  Epochs: {num_epochs} (DOUBLED for better convergence)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (INCREASED for faster learning)")
    print(f"  Optimizer: AdamW (reduced weight decay)")
    print(f"  Scheduler: CosineAnnealingLR with warmup")
    print(f"  Dataset size: {len(train_dataset)} train, {len(val_dataset)} val (5x LARGER)")
    print(f"  Model: Optimized ViT-B/32 (reduced depth for efficiency)")
    print(f"  Temperature: 0.05 (LOWER for sharper alignment)")
    
    # Training loop with EARLY STOPPING
    print(f"\n🚀 Starting ENHANCED training...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 15  # Early stopping if no improvement for 15 epochs

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
            torch.save(model.state_dict(), 'models/clip_mnist_best.pth')
            print("✅ Best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/clip_mnist_final.pth')
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
