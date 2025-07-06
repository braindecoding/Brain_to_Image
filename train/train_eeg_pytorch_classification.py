#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegclassifier_pytorch import convolutional_encoder_model

def eeg_data_augmentation(x, y, augment_factor=2):
    """
    Apply data augmentation techniques for EEG data
    """
    augmented_x = []
    augmented_y = []

    for i in range(len(x)):
        # Original data
        augmented_x.append(x[i])
        augmented_y.append(y[i])

        # Augmentation techniques
        for _ in range(augment_factor):
            sample = x[i].copy()

            # 1. Add small amount of Gaussian noise (0.5% of signal std)
            noise_std = np.std(sample) * 0.005
            sample += np.random.normal(0, noise_std, sample.shape)

            # 2. Small time shift (±1 sample)
            shift = np.random.randint(-1, 2)
            if shift != 0:
                sample = np.roll(sample, shift, axis=-1)

            # 3. Small amplitude scaling (±2%)
            scale = np.random.uniform(0.98, 1.02)
            sample *= scale

            augmented_x.append(sample)
            augmented_y.append(y[i])

    return np.array(augmented_x), np.array(augmented_y)

def load_and_prepare_data():
    """Load and prepare EEG data - EXACT SAME as Keras version"""
    print("=== Loading Data (Exact Keras Match) ===")
    
    # Updated paths to match actual data location
    run_id = "eeg_classifier_adm5"
    dataset = "MNIST_EP"
    root_dir = "data"  # Data is in data/ folder
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
    
    # Apply MODERATE data augmentation to training data only
    print("Applying MODERATE EEG data augmentation to training data...")
    x_train_aug, y_train_aug = eeg_data_augmentation(x_train, y_train, augment_factor=2)  # MODERATE augmentation
    print(f"Augmented training data: {x_train.shape} -> {x_train_aug.shape}")

    # Keep labels in one-hot format like successful Keras version
    print("Keeping labels in one-hot format (same as successful Keras)")

    # Convert to tensors
    x_train = torch.FloatTensor(x_train_aug)
    y_train = torch.FloatTensor(y_train_aug)  # One-hot labels as float
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)    # One-hot labels as float
    
    # Create validation split - EXACT SAME as Keras (validation_split=0.25)
    train_size = int(0.75 * len(x_train))
    indices = torch.randperm(len(x_train))
    
    x_train_split = x_train[indices[:train_size]]
    y_train_split = y_train[indices[:train_size]]
    x_val_split = x_train[indices[train_size:]]
    y_val_split = y_train[indices[train_size:]]
    
    print(f"After validation split:")
    print(f"  x_train: {x_train_split.shape}")
    print(f"  y_train: {y_train_split.shape} (dtype: {y_train_split.dtype})")
    print(f"  x_val: {x_val_split.shape}")
    print(f"  y_val: {y_val_split.shape} (dtype: {y_val_split.dtype})")
    print(f"  x_test: {x_test.shape}")
    print(f"  y_test: {y_test.shape} (dtype: {y_test.dtype})")
    print(f"  y_train sample: {y_train_split[:5]}")
    print(f"  y_train unique: {torch.unique(y_train_split)}")
    
    return x_train_split, y_train_split, x_val_split, y_val_split, x_test, y_test

def train_model():
    """Main training function - EXACT SAME as Keras version"""
    print("🧠 EEG Classification Training (PyTorch - Exact Keras Match)")
    print("=" * 70)
    
    # Set device with GPU support and cuDNN optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"� Using GPU: {torch.cuda.get_device_name()}")

        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Set memory growth to avoid OOM
        torch.cuda.empty_cache()

    else:
        device = torch.device('cpu')
        print("🖥️ Using CPU (GPU not available)")

    print(f"Device: {device}")
    
    try:
        # Load data
        x_train, y_train, x_val, y_val, x_test, y_test = load_and_prepare_data()
        
        # Get data dimensions - EXACT SAME as Keras
        channels = x_train.shape[2]  # x_train.shape[1] in Keras
        observations = x_train.shape[3]  # x_train.shape[2] in Keras
        
        print(f"Model input dimensions:")
        print(f"  Channels: {channels}")
        print(f"  Observations: {observations}")
        
        # Create model - EXACT SAME as Keras
        print("=== Creating Model ===")
        # Create model WITH softmax like Keras (for one-hot label compatibility)
        classifier = convolutional_encoder_model(channels, observations, 10, verbose=True, use_softmax=True)
        classifier = classifier.to(device)
        
        # BALANCED REGULARIZATION training parameters
        batch_size = 32   # OPTIMAL batch size (same as successful Keras)
        num_epochs = 700  # EXTENDED epochs to reach 85% target

        # MODERATE REGULARIZATION Adam optimizer
        # Balanced weight decay for good generalization without underfitting
        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)

        # MODERATE loss function with balanced label smoothing
        def categorical_crossentropy_with_label_smoothing(y_pred, y_true, smoothing=0.1):
            """Categorical crossentropy with MODERATE label smoothing for balanced learning"""
            # Apply softmax to predictions
            y_pred_softmax = torch.softmax(y_pred, dim=1)

            # Apply moderate label smoothing
            num_classes = y_true.shape[1]
            y_true_smooth = y_true * (1 - smoothing) + smoothing / num_classes

            # Compute categorical crossentropy
            loss = -torch.sum(y_true_smooth * torch.log(y_pred_softmax + 1e-8), dim=1)
            return torch.mean(loss)

        criterion = lambda pred, true: categorical_crossentropy_with_label_smoothing(pred, true, smoothing=0.1)

        # MODERATE learning rate reduction for balanced learning
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=20, verbose=True, min_lr=1e-5)
        
        # Create data loaders
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup model saving - Updated for correct path
        model_save_dir = "models"  # Save models in main models/ folder
        os.makedirs(model_save_dir, exist_ok=True)

        run_id = "eeg_classifier_adm5"
        saved_model_file = os.path.join(model_save_dir, str(run_id) + '_final.pth')
        best_model_file = os.path.join(model_save_dir, str(run_id) + '_best.pth')

        # Training variables with BALANCED early stopping
        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 50  # MODERATE early stopping for balanced learning
        train_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        print(f"=== Starting BALANCED REGULARIZATION Training ===")
        print(f"Batch size: {batch_size} (OPTIMAL size matching successful Keras)")
        print(f"Epochs: {num_epochs} (EXTENDED to reach 85% target)")
        print(f"Optimizer: Adam with BALANCED regularization")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']} (RESTORED for good learning)")
        print(f"Weight decay: 5e-4 (MODERATE regularization)")
        print(f"Data augmentation: 3x training data (BALANCED noise + time shift + scaling)")
        print(f"Label smoothing: 0.1 (MODERATE smoothing for balanced learning)")
        print(f"Early stopping patience: {early_stop_patience} epochs (BALANCED patience)")
        print(f"LR scheduler: factor=0.7, patience=20 (MODERATE reduction)")
        print(f"Min LR: 1e-5 (REASONABLE minimum)")
        print(f"Model save dir: {model_save_dir}")
        print(f"Expected training time: ~{num_epochs * 0.1:.1f} minutes on GPU")
        print(f"🎯 Target: Train 90-95%, Val 80-85%, Gap <10%")
        print(f"🔧 BALANCED techniques: Moderate Reg + Balanced Aug + Smart Early Stop!")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output, _ = classifier(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # For one-hot labels, get predictions and true labels
                pred = output.argmax(dim=1)
                true_labels = target.argmax(dim=1)
                train_correct += pred.eq(true_labels).sum().item()
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
                    # For one-hot labels, get predictions and true labels
                    pred = output.argmax(dim=1)
                    true_labels = target.argmax(dim=1)
                    val_correct += pred.eq(true_labels).sum().item()
                    val_total += target.size(0)
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save best model (EXACT SAME as Keras ModelCheckpoint)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0  # Reset patience counter
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss,
                }, best_model_file)
                print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
            
            # Store history
            train_history['loss'].append(avg_train_loss)
            train_history['accuracy'].append(train_acc)
            train_history['val_loss'].append(avg_val_loss)
            train_history['val_accuracy'].append(val_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.8f}")
            print(f"Patience: {patience_counter}/{early_stop_patience}")

            # Warning if LR gets too low
            if current_lr < 1e-6:
                print(f"⚠️  WARNING: LR very low ({current_lr:.2e}), may need adjustment")
            if current_lr < 1e-7:
                print(f"🚨 CRITICAL: LR extremely low, consider stopping or adjusting")

            # Save checkpoint every 50 epochs and show progress
            if (epoch + 1) % 50 == 0:
                checkpoint_file = os.path.join(model_save_dir, f"{run_id}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss,
                }, checkpoint_file)
                print(f"📁 Checkpoint saved: {checkpoint_file}")
                print(f"🎯 Progress: {(epoch+1)/num_epochs*100:.1f}% complete")
                print(f"📈 Best so far: {best_val_acc:.2f}% (target: >85%)")

            # Show detailed progress every 25 epochs
            if (epoch + 1) % 25 == 0:
                train_val_gap = train_acc - val_acc
                print(f"📊 Train-Val Gap: {train_val_gap:.2f}% (target: <15%)")
                print(f"📈 Gap trend: Previous best gap was ~26%, current: {train_val_gap:.2f}%")
                if train_val_gap > 25:
                    print("🔥 VERY LARGE gap - need more regularization or training!")
                elif train_val_gap > 20:
                    print("🔥 LARGE gap - significant room for improvement")
                elif train_val_gap > 15:
                    print("⚠️  Moderate gap - getting better")
                elif train_val_gap > 10:
                    print("✅ Good gap - approaching target")
                else:
                    print("🎯 Excellent gap - near optimal generalization")

                # LR status
                if current_lr > 1e-4:
                    print(f"💪 LR still healthy: {current_lr:.6f}")
                else:
                    print(f"⚠️  LR getting low: {current_lr:.6f} - may limit improvement")

            # Ultra-detailed monitoring every 100 epochs
            if (epoch + 1) % 100 == 0:
                improvement_rate = (val_acc - best_val_acc) if epoch > 100 else val_acc
                print(f"🔍 DEEP ANALYSIS at epoch {epoch+1}:")
                print(f"   📈 Recent improvement: {improvement_rate:.3f}%")
                print(f"   🎯 Distance to 85% target: {85 - val_acc:.2f}%")
                print(f"   ⏱️  Time elapsed: ~{(epoch+1) * 0.1:.1f} minutes")
                print(f"   🔋 Patience remaining: {early_stop_patience - patience_counter}")
                if val_acc > 85:
                    print("🎉 TARGET ACHIEVED! But continuing for maximum...")
                elif val_acc > 82:
                    print("🔥 VERY CLOSE to target! Keep pushing!")
                elif val_acc > 80:
                    print("💪 Good progress toward target!")
                else:
                    print("🚀 Still climbing toward target!")
        
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
                # For one-hot labels, get predictions and true labels
                pred = output.argmax(dim=1)
                true_labels = target.argmax(dim=1)
                test_correct += pred.eq(true_labels).sum().item()
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
