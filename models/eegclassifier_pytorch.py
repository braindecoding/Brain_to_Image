import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

"""
Title: EEG Classifier model design (PyTorch Version)

Purpose:
    PyTorch implementation of EEG Classifier, Functional blocks for training
    CNN Classifier model. Converted from Keras version.

Author: Tim Tanner (Original), Converted to PyTorch
Date: 01/07/2024
Version: PyTorch v1.0

Usage:
    Build CNN model using PyTorch

Notes:
    - Converted from Keras Sequential model to PyTorch nn.Module
    - Maintains same architecture and functionality
    - Added proper weight initialization
    - Compatible with PyTorch training loops

Examples:
    model = EEGClassifier(channels=9, observations=32, num_classes=10)
    output = model(input_tensor)

    # Exact same as Keras version
model = convolutional_encoder_model(9, 32, 10, verbose=True)

# Or with PyTorch class directly
model = EEGClassifier(9, 32, 10, dropout_rate=0.1, l2_reg=0.015, use_softmax=True)
"""

class EEGClassifier(nn.Module):
    """
    PyTorch implementation of EEG Classifier CNN
    
    Architecture:
    - BatchNorm -> Conv2D layers -> MaxPool -> Dense layers -> Output
    - Designed for EEG signal classification to 10 digit classes
    - Extracts 128-dim latent features for GAN input
    """
    
    def __init__(self, channels, observations, num_classes=10, dropout_rate=0.1, l2_reg=0.015, use_softmax=False):
        super(EEGClassifier, self).__init__()
        
        self.channels = channels
        self.observations = observations
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_softmax = use_softmax
        
        # Input shape: (batch_size, 1, channels, observations)
        # Note: PyTorch uses (N, C, H, W) format
        
        # Initial Batch Normalization
        self.bn1 = nn.BatchNorm2d(1)
        
        # Convolutional layers - EXACT MATCH with Keras version
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, 4), padding='same')  # EEG_series_Conv2D
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(channels, 1), padding='same')  # EEG_channel_Conv2D
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # EEG_feature_pool1

        # IMPORTANT: These kernel sizes match the Keras original exactly
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(4, 25), padding='same')  # EEG_feature_Conv2D1
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))  # EEG_feature_pool2

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(50, 2), padding='same')  # EEG_feature_Conv2D2
        
        # Calculate flattened size after convolutions
        # This will be computed dynamically in forward pass
        self.flatten_size = None
        
        # Dense layers
        self.bn2 = nn.BatchNorm1d(512)  # Will be adjusted based on actual flatten size
        self.fc1 = nn.Linear(self.flatten_size or 1024, 512)  # EEG_feature_FC512
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)  # EEG_feature_FC256
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)  # EEG_feature_FC128 (latent space)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.bn3 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, num_classes)  # EEG_Class_Labels (L2 reg applied in optimizer)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, channels, observations)
            
        Returns:
            output: Classification logits (batch_size, num_classes)
            features: 128-dim latent features (batch_size, 128)
        """
        # Input batch normalization
        x = self.bn1(x)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv4(x))
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Update flatten size if not set
        if self.flatten_size is None:
            self.flatten_size = x.size(1)
            # Reinitialize first linear layer with correct input size
            self.fc1 = nn.Linear(self.flatten_size, 512).to(x.device)
            self.bn2 = nn.BatchNorm1d(512).to(x.device)
        
        # Dense layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Latent features (128-dim for GAN)
        features = self.fc3(x)
        features = F.relu(features)
        features = self.dropout3(features)
        features = self.bn3(features)
        
        # Classification output
        output = self.fc_out(features)

        # Apply softmax if requested (for compatibility with Keras)
        if self.use_softmax:
            output = F.softmax(output, dim=1)

        return output, features
    
    def get_latent_features(self, x):
        """
        Extract 128-dim latent features for GAN input
        
        Args:
            x: Input tensor
            
        Returns:
            features: 128-dim latent features
        """
        _, features = self.forward(x)
        return features
    
    def summary(self):
        """Print model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 80)
        print(f"EEG Classifier (PyTorch)")
        print("=" * 80)
        print(f"Input shape: (batch_size, 1, {self.channels}, {self.observations})")
        print(f"Output classes: {self.num_classes}")
        print(f"Latent features: 128")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 80)
        
        # Print layer details
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                print(f"{name:30} {str(module):50} {params:>10,}")
        print("=" * 80)


def convolutional_encoder_model(channels, observations, num_classes, verbose=False):
    """
    Factory function to create EEG classifier (for compatibility with Keras version)
    
    Args:
        channels: Number of EEG channels
        observations: Number of time observations
        num_classes: Number of output classes
        verbose: Whether to print model summary
        
    Returns:
        model: EEGClassifier instance
    """
    model = EEGClassifier(channels, observations, num_classes)
    
    if verbose:
        model.summary()
    
    return model


class EEGClassifierSimple(nn.Module):
    """
    Simplified version of EEG Classifier for better stability
    """
    
    def __init__(self, channels, observations, num_classes=10):
        super(EEGClassifierSimple, self).__init__()
        
        self.channels = channels
        self.observations = observations
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate flatten size
        self.flatten_size = self._get_flatten_size()
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)  # Latent features
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc_out = nn.Linear(128, num_classes)
    
    def _get_flatten_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.channels, self.observations)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        features = F.relu(self.fc2(x))
        features = self.dropout2(features)
        
        output = self.fc_out(features)
        
        return output, features


if __name__ == '__main__':
    # Test the model
    print("Testing EEG Classifier (PyTorch)")
    
    # Create model
    model = convolutional_encoder_model(9, 32, 10, verbose=True)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 9, 32)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    with torch.no_grad():
        output, features = model(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    print("\n✅ PyTorch EEG Classifier test completed successfully!")
