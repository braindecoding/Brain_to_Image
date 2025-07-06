#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegclassifier_pytorch import convolutional_encoder_model

def load_and_prepare_data():
    """Load and prepare EEG data - EXACT SAME as Keras version"""
    print("=== Loading Data (Exact Keras Match) ===")
    
    # EXACT SAME paths as Keras version
    run_id = "eeg_classifier_adm5"
    dataset = "MNIST_EP"
    root_dir = f"Datasets/MindBigData MNIST of Brain Digits/{dataset}"
    data_file = "data_train_MindBigData2022_MNIST_EP.pkl"
    
    print(f"Reading data file {root_dir}/{data_file}")
    
    if not os.path.exists(f"{root_dir}/{data_file}"):
        raise FileNotFoundError(f"Data file not found: {root_dir}/{data_file}")
    
    # Load data - EXACT SAME as Keras
    eeg_data = pickle.load(open(f"{root_dir}/{data_file}", 'rb'), encoding='bytes')
    
    # Extract data - EXACT SAME as Keras
    x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_val'], eeg_data['y_val']
    
    print(f"Original shapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Convert to PyTorch format (N, C, H, W)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    elif len(x_train.shape) == 4 and x_train.shape[-1] == 1:
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))
    
    # Convert to tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    
    # Create validation split - EXACT SAME as Keras (validation_split=0.25)
    train_size = int(0.75 * len(x_train))
    indices = torch.randperm(len(x_train))
    
    x_train_split = x_train[indices[:train_size]]
    y_train_split = y_train[indices[:train_size]]
    x_val_split = x_train[indices[train_size:]]
    y_val_split = y_train[indices[train_size:]]
    
    print(f"After validation split:")
    print(f"  x_train: {x_train_split.shape}")
    print(f"  y_train: {y_train_split.shape}")
    print(f"  x_val: {x_val_split.shape}")
    print(f"  y_val: {y_val_split.shape}")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return x_train_split, y_train_split, x_val_split, y_val_split, x_test, y_test, root_dir

def train_model():
    """Main training function - EXACT SAME as Keras version"""
    print("🧠 EEG Classification Training (PyTorch - Exact Keras Match)")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load data
        x_train, y_train, x_val, y_val, x_test, y_test, root_dir = load_and_prepare_data()
        
        # Get data dimensions - EXACT SAME as Keras
        channels = x_train.shape[2]  # x_train.shape[1] in Keras
        observations = x_train.shape[3]  # x_train.shape[2] in Keras
        
        print(f"Model input dimensions:")
        print(f"  Channels: {channels}")
        print(f"  Observations: {observations}")
        
        # Create model - EXACT SAME as Keras
        print("=== Creating Model ===")
        classifier = convolutional_encoder_model(channels, observations, 10, verbose=True)
        classifier = classifier.to(device)
        
        # EXACT SAME training parameters as Keras
        batch_size = 128  # SAME as Keras
        num_epochs = 150  # SAME as Keras
        
        # EXACT SAME optimizer as Keras
        # Keras: Adam(learning_rate=0.0001, beta_1=0.9, decay=1e-6)
        optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-6)
        
        # EXACT SAME loss function as Keras
        criterion = nn.CrossEntropyLoss()
        
        # EXACT SAME learning rate scheduler as Keras
        # Keras: ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
        
        # Create data loaders
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup model saving - EXACT SAME as Keras
        model_save_dir = os.path.join(root_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        
        run_id = "eeg_classifier_adm5"
        saved_model_file = os.path.join(model_save_dir, str(run_id) + '_final.pth')
        best_model_file = os.path.join(model_save_dir, str(run_id) + '_best.pth')
        
        print(f"=== Starting Training ===")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Model save dir: {model_save_dir}")
        
        # Training variables
        best_val_acc = 0
        train_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output, _ = classifier(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            classifier.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output, _ = classifier(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save best model (EXACT SAME as Keras ModelCheckpoint)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss,
                }, best_model_file)
                print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%")
            
            # Store history
            train_history['loss'].append(avg_train_loss)
            train_history['accuracy'].append(train_acc)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_accuracy'].append(val_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save final model and history - EXACT SAME as Keras
        torch.save(classifier.state_dict(), saved_model_file)
        
        # Save training history - EXACT SAME as Keras
        history_file = os.path.join(model_save_dir, f"history_{run_id}_final.pkl")
        with open(history_file, "wb") as f:
            pickle.dump(train_history, f)
        
        print(f"Model saved to: {saved_model_file}")
        print(f"History saved to: {history_file}")
        
        # Final evaluation on test set - EXACT SAME as Keras
        classifier.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = classifier(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\n=== Final Results ===")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print([avg_test_loss, test_acc/100])  # EXACT SAME format as Keras
        
        print(f"\n🎉 Training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n✅ PyTorch EEG Classification training completed (Exact Keras Match)!")
    else:
        print("\n❌ Training failed!")
    
    sys.exit(0 if success else 1)
