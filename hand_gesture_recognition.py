import os
import sys
import glob
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ---------------------------------------------------------
# Helper: Clean directory names to pretty class labels
# ---------------------------------------------------------
def clean_class_name(dir_name):
    """
    Cleans raw gesture directory names like '01_palm' or '02_l'
    into highly readable class labels like 'Palm' or 'L-Gesture'.
    """
    parts = dir_name.split('_')
    if len(parts) > 1:
        name = ' '.join(parts[1:])
        if name.lower() == 'l':
            return 'L-Gesture'
        return name.title()
    return dir_name.title()

# ---------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------
class GestureDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Discover directories that contain PNG images
        self.class_dirs = []
        for root, dirs, files in os.walk(base_dir):
            png_files = [f for f in files if f.lower().endswith('.png')]
            if png_files:
                self.class_dirs.append(root)
                
        self.class_dirs = sorted(self.class_dirs)
        self.class_names = [os.path.basename(d) for d in self.class_dirs]
        self.clean_names = [clean_class_name(name) for name in self.class_names]
        
        print("\n[Data Discovery] Found the following gesture directories:")
        for idx, (raw, clean) in enumerate(zip(self.class_names, self.clean_names)):
            print(f"  Class {idx:02d}: '{raw}' -> mapped to '{clean}'")
            
        # Collect paths and labels
        for class_idx, class_dir in enumerate(self.class_dirs):
            png_paths = glob.glob(os.path.join(class_dir, "*.png"))
            self.image_paths.extend(png_paths)
            self.labels.extend([class_idx] * len(png_paths))
            
        print(f"[Dataset Stats] Total size: {len(self.image_paths)} images across {len(self.class_dirs)} classes.")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Grayscale images loaded as mode L
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"  [Warning] Failed to load image {img_path}: {e}")
            image = Image.new('L', (128, 128), 0)
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        return image, label

# ---------------------------------------------------------
# CNN Model Architecture
# ---------------------------------------------------------
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        # Input: 1 x 128 x 128 (Grayscale)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 16 x 64 x 64
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 32 x 32 x 32
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 64 x 16 x 16
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_block(x)
        return x

# ---------------------------------------------------------
# Training Loop & Plotting
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, device, num_classes, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print("\n" + "="*50)
    print("                STARTING MODEL TRAINING")
    print("="*50)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        # Save historical metrics
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
              
        # Checkpointing: save best validation model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_gesture_model.pth')
            print("  --> Best model checkpoint saved to 'best_gesture_model.pth'")
            
    total_time = time.time() - start_time
    print("="*50)
    print(f"Training completed in {total_time:.2f} seconds.")
    print("="*50)
    
    # Save training curves as a beautiful visual plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
    plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Subplot 2: Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='Train Acc', color='#2ecc71', linewidth=2)
    plt.plot(range(1, epochs+1), history['val_acc'], label='Val Acc', color='#f1c40f', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("[Success] Training history curves saved to 'training_history.png'.")

# ---------------------------------------------------------
# Evaluation Module
# ---------------------------------------------------------
def evaluate_model(model, test_loader, device, class_names):
    print("\n" + "-"*50)
    print("             MODEL EVALUATION REPORT")
    print("-"*50)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate performance metrics
    print("\nClassification Metrics:")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot standard modern confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Render values in each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11)
                 
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("[Success] Confusion matrix visualization saved to 'confusion_matrix.png'.")

# ---------------------------------------------------------
# Random Prediction Visualization
# ---------------------------------------------------------
def test_random_predictions(model, device, test_dataset, class_names):
    print("\n--- Running Inference on Random Test Images ---")
    if len(test_dataset) == 0:
        print("[Error] No test images available.")
        return
        
    num_samples = min(6, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    model.eval()
    for i, idx in enumerate(indices):
        img_tensor, label_idx = test_dataset[idx]
        
        # Prepare batch input
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        pred_label = class_names[predicted_idx.item()]
        true_label = class_names[label_idx]
        conf_score = confidence.item()
        
        # Convert tensor to numpy for plotting
        img_np = img_tensor.squeeze().numpy()
        
        axes[i].imshow(img_np, cmap='gray')
        
        # Green if accurate, red if mismatch
        color = '#2ecc71' if predicted_idx.item() == label_idx else '#e74c3c'
        
        title_str = f"True: {true_label}\nPred: {pred_label} ({conf_score*100:.1f}%)"
        axes[i].set_title(title_str, color=color, fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    output_path = 'test_predictions.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Success] Inference plots successfully saved to '{output_path}'.")

# ---------------------------------------------------------
# Interactive Live Webcam Demo
# ---------------------------------------------------------
def webcam_demo(model, device, class_names):
    print("\n" + "="*50)
    print("            LAUNCHING LIVE WEBCAM DEMO")
    print("="*50)
    print("   [Instruction] Align your hand inside the GREEN box.")
    print("   [Instruction] Press 'q' inside the webcam window to exit.")
    print("-"*50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[Error] Unable to initialize webcam.")
        print("Please check that a working camera is connected to your system")
        print("and that no other application is currently using it.")
        return
        
    cv2.namedWindow("Live Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Gesture Recognition", 800, 600)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to capture frames from webcam.")
                break
                
            # Flip image horizontally for a natural mirror view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Crop a central square Region of Interest (ROI)
            roi_size = int(min(h, w) * 0.6)
            start_y = (h - roi_size) // 2
            start_x = (w - roi_size) // 2
            end_y = start_y + roi_size
            end_x = start_x + roi_size
            
            # Extract ROI
            roi = frame[start_y:end_y, start_x:end_x]
            
            # --- Software Fix: Background Removal using Skin Masking ---
            # Blur to reduce noise before skin detection
            blurred_roi = cv2.GaussianBlur(roi, (7, 7), 0)
            
            # Convert to HSV color space for skin detection
            hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
            
            # HSV skin range (tight thresholds to avoid detecting background)
            lower_skin = np.array([0, 40, 80], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Second range to catch reddish skin tones that wrap around hue=180
            lower_skin2 = np.array([160, 40, 80], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            # Combine both ranges
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Strong morphological cleanup to remove background noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # --- Replace the ROI area: infrared-style (black bg + white hand) ---
            # Convert ROI to grayscale to mimic infrared sensor look
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Apply the mask: keep only the hand, everything else becomes black
            ir_style = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
            # Boost brightness of the hand to make it glow white like IR data
            ir_style = cv2.normalize(ir_style, None, 0, 255, cv2.NORM_MINMAX)
            # Convert back to 3-channel so it can be placed into the BGR frame
            ir_colored = cv2.cvtColor(ir_style, cv2.COLOR_GRAY2BGR)
            frame[start_y:end_y, start_x:end_x] = ir_colored
            
            # Preprocess ROI (resize the binary mask to 128x128)
            # This perfectly mimics the bright-hand/dark-background of the Leap IR dataset!
            resized_mask = cv2.resize(mask, (128, 128))
            
            # Normalization and Tensor formatting
            input_tensor = torch.tensor(resized_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, dim=0)
                
            predicted_class = class_names[predicted_idx.item()]
            confidence_pct = confidence.item() * 100
            
            # Render green border around the mask area
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (46, 204, 113), 3) # Green box
            
            # Render predicted label & UI backdrop box
            ui_text = f"Predicted: {predicted_class} ({confidence_pct:.1f}%)"
            
            # Dark backing banner for crisp text contrast
            cv2.rectangle(frame, (0, 0), (w, 60), (44, 62, 80), -1)
            cv2.putText(frame, ui_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (236, 240, 241), 2, cv2.LINE_AA)
            
            # Overlay instructions
            cv2.putText(frame, "Align Hand inside green square", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to Quit", (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (231, 76, 60), 2, cv2.LINE_AA)
            
            cv2.imshow("Live Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"\n[Runtime Error in Demo] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nLive Webcam Demo stopped and hardware resources released.")

# ---------------------------------------------------------
# Interactive Main Command Interface
# ---------------------------------------------------------
def print_banner():
    banner = """
==============================================================
   _    _                 _  _____           _ 
  | |  | |               | |/ ____|         | |
  | |__| | __ _ _ __   __| | |  __  ___  ___| |_ _   _ _ __ ___
  |  __  |/ _` | '_ \ / _` | | |_ |/ _ \/ __| __| | | | '__/ _ \\
  | |  | | (_| | | | | (_| | |__| |  __/\\__ \\ |_| |_| | | |  __/
  |_|  |_|\\__,_|_| |_|\\__,_|\\_____|\\___||___/\\__|\\__,_|_|  \\___|
        H A N D   G E S T U R E   R E C O G N I T I O N
==============================================================
    Built with PyTorch & OpenCV | Modern Neural Network
==============================================================
"""
    print(banner)

def main():
    print_banner()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device info] Executing computations on: {device.type.upper()}")
    
    # Path configuration
    dataset_path = 'leapGestRecog/00'
    if not os.path.exists(dataset_path):
        print(f"\n[Fatal Error] Dataset folder not found at path: {dataset_path}")
        print("Please check that the 'leapGestRecog/00' directory is present.")
        sys.exit(1)
        
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Discover and build full dataset mapping
    print("Scanning dataset directory...")
    full_dataset = GestureDataset(dataset_path)
    class_names = full_dataset.clean_names
    num_classes = len(class_names)
    
    # Split training and testing sets
    generator = torch.Generator().manual_seed(42)
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    train_subset, test_subset = random_split(full_dataset, [train_len, test_len], generator=generator)
    
    class SubDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            orig_idx = self.subset.indices[idx]
            image_path = self.subset.dataset.image_paths[orig_idx]
            label = self.subset.dataset.labels[orig_idx]
            
            image = Image.open(image_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label
            
    train_data = SubDataset(train_subset, train_transform)
    test_data = SubDataset(test_subset, val_transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Define Model
    model = GestureCNN(num_classes=num_classes).to(device)
    model_loaded = False
    
    # Auto-load weights if they exist
    weights_file = 'best_gesture_model.pth'
    if os.path.exists(weights_file):
        try:
            model.load_state_dict(torch.load(weights_file, map_location=device))
            model_loaded = True
            print(f"\n[Model Info] Successfully loaded pre-trained model weights from '{weights_file}'.")
        except Exception as e:
            print(f"\n[Warning] Found '{weights_file}' but failed to load weights: {e}")
            
    while True:
        print("\n" + "="*40)
        print("               SYSTEM MENU")
        print("="*40)
        print("  1. Train Model (Runs PyTorch CNN on CPU)")
        print("  2. Evaluate Model (Generates classification report)")
        print("  3. Run Random Predictions Visualizer")
        print("  4. Launch Live Webcam Demo (Requires camera)")
        print("  5. Exit")
        print("-"*40)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            train_model(model, train_loader, test_loader, device, num_classes, epochs=10)
            model_loaded = True
            if os.path.exists(weights_file):
                model.load_state_dict(torch.load(weights_file, map_location=device))
                
        elif choice == '2':
            if not model_loaded:
                print("\n[Warning] Please train the model first (Option 1) to load/save weights.")
                continue
            evaluate_model(model, test_loader, device, class_names)
            
        elif choice == '3':
            if not model_loaded:
                print("\n[Warning] Please train the model first (Option 1) to load/save weights.")
                continue
            test_random_predictions(model, device, test_data, class_names)
            
        elif choice == '4':
            if not model_loaded:
                print("\n[Warning] Please train the model first (Option 1) to load/save weights.")
                continue
            webcam_demo(model, device, class_names)
            
        elif choice == '5':
            print("\nExiting Hand Gesture Recognition system. Goodbye!")
            break
        else:
            print("\n[Warning] Invalid choice. Please enter a number between 1 and 5.")

if __name__ == '__main__':
    main()
