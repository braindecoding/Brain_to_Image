#!/usr/bin/env python3

"""
EEG Model Inference Utility

Purpose:
    Load trained EEG classifier and perform inference
    Used in CLIP training pipeline for EEG → digit prediction

Author: Brain-to-Image Pipeline
Date: 07/07/2024
Version: v1.0

Usage:
    inferencer = EEGInferencer('models/eeg_classifier_adm5_final.pth')
    predictions = inferencer.predict(eeg_data)
    probabilities = inferencer.predict_proba(eeg_data)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegclassifier_pytorch import convolutional_encoder_model

class EEGInferencer:
    """
    EEG Model Inference Class
    
    Loads trained EEG classifier and provides inference methods
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize EEG inferencer
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_path = model_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🧠 EEG Inferencer initialized on device: {self.device}")
        
        # Load model
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained EEG classifier model"""
        try:
            print(f"📂 Loading EEG model from: {self.model_path}")
            
            # Create model architecture (same as training)
            # EEG data dimensions: 9 channels, 32 observations, 10 classes
            self.model = convolutional_encoder_model(
                channels=9, 
                observations=32, 
                num_classes=10, 
                verbose=False,
                use_softmax=False  # We'll apply softmax manually
            )
            
            # Load weights with proper handling of dynamic flatten size
            if os.path.exists(self.model_path):
                # Move model to device first
                self.model = self.model.to(self.device)

                # Set model to eval mode first to avoid BatchNorm issues
                self.model.eval()

                # Handle dynamic flatten size by doing a forward pass first
                dummy_input = torch.randn(2, 1, 9, 32).to(self.device)  # Use batch size 2 to avoid BatchNorm error
                with torch.no_grad():
                    _ = self.model(dummy_input)  # This will set the correct flatten_size

                # Load state dict
                state_dict = torch.load(self.model_path, map_location=self.device)

                # Try to load, if size mismatch, handle it
                try:
                    self.model.load_state_dict(state_dict)
                    print("✅ Model weights loaded successfully!")
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        print("⚠️ Size mismatch detected, loading compatible weights...")
                        # Load only compatible weights
                        model_dict = self.model.state_dict()
                        compatible_dict = {}
                        for k, v in state_dict.items():
                            if k in model_dict and model_dict[k].shape == v.shape:
                                compatible_dict[k] = v
                            else:
                                print(f"  Skipping {k}: shape mismatch {model_dict.get(k, 'missing').shape if k in model_dict else 'missing'} vs {v.shape}")

                        model_dict.update(compatible_dict)
                        self.model.load_state_dict(model_dict)
                        print("✅ Compatible weights loaded successfully!")
                    else:
                        raise e
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_eeg_data(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess EEG data for inference
        
        Args:
            eeg_data: EEG data with shape (batch_size, channels, observations) or 
                     (channels, observations) for single sample
        
        Returns:
            preprocessed_data: Tensor ready for model input
        """
        # Convert to tensor if numpy
        if isinstance(eeg_data, np.ndarray):
            eeg_data = torch.FloatTensor(eeg_data)
        
        # Add batch dimension if single sample
        if len(eeg_data.shape) == 2:
            eeg_data = eeg_data.unsqueeze(0)  # (channels, observations) -> (1, channels, observations)
        
        # Add channel dimension for PyTorch conv2d: (batch, 1, channels, observations)
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.unsqueeze(1)  # (batch, channels, observations) -> (batch, 1, channels, observations)
        
        # Move to device
        eeg_data = eeg_data.to(self.device)
        
        return eeg_data
    
    def predict(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict digit classes from EEG data
        
        Args:
            eeg_data: EEG data
            
        Returns:
            predictions: Predicted digit classes (0-9)
        """
        with torch.no_grad():
            # Preprocess data
            processed_data = self.preprocess_eeg_data(eeg_data)
            
            # Forward pass
            logits, features = self.model(processed_data)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            return predictions.cpu().numpy()
    
    def predict_proba(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities from EEG data
        
        Args:
            eeg_data: EEG data
            
        Returns:
            probabilities: Class probabilities for each digit (0-9)
        """
        with torch.no_grad():
            # Preprocess data
            processed_data = self.preprocess_eeg_data(eeg_data)
            
            # Forward pass
            logits, features = self.model(processed_data)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            return probabilities.cpu().numpy()
    
    def extract_features(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract 128-dim latent features from EEG data
        
        Args:
            eeg_data: EEG data
            
        Returns:
            features: 128-dimensional feature vectors
        """
        with torch.no_grad():
            # Preprocess data
            processed_data = self.preprocess_eeg_data(eeg_data)
            
            # Forward pass
            logits, features = self.model(processed_data)
            
            return features.cpu().numpy()
    
    def predict_with_confidence(self, eeg_data: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores
        
        Args:
            eeg_data: EEG data
            
        Returns:
            predictions: Predicted classes
            confidences: Confidence scores (max probability)
        """
        probabilities = self.predict_proba(eeg_data)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences


def test_eeg_inferencer():
    """Test function for EEG inferencer"""
    print("Testing EEGInferencer...")
    
    # Check if model exists
    model_path = "models/eeg_classifier_adm5_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the EEG model first!")
        return
    
    try:
        # Initialize inferencer
        inferencer = EEGInferencer(model_path)
        
        # Create dummy EEG data
        print("\n=== Testing with dummy EEG data ===")
        
        # Single sample test
        single_eeg = np.random.randn(9, 32)  # 9 channels, 32 observations
        pred = inferencer.predict(single_eeg)
        proba = inferencer.predict_proba(single_eeg)
        features = inferencer.extract_features(single_eeg)
        
        print(f"Single sample prediction: {pred[0]}")
        print(f"Probabilities shape: {proba.shape}")
        print(f"Features shape: {features.shape}")
        
        # Batch test
        batch_eeg = np.random.randn(5, 9, 32)  # 5 samples
        batch_pred = inferencer.predict(batch_eeg)
        batch_proba = inferencer.predict_proba(batch_eeg)
        
        print(f"\nBatch predictions: {batch_pred}")
        print(f"Batch probabilities shape: {batch_proba.shape}")
        
        # Confidence test
        pred_conf, conf_scores = inferencer.predict_with_confidence(batch_eeg)
        print(f"Predictions with confidence: {list(zip(pred_conf, conf_scores))}")
        
        print("\n✅ EEG inferencer test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_eeg_inferencer()
