import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from scipy.io import loadmat, savemat
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a custom head for augmentation, used in the model
class AugmentHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.augment_head = nn.Linear(dim, 12)  # Output layer with 12 features

    def forward(self, x):
        x = self.augment_head(x)  # Pass input through the augmentation head
        return x
    
# Load a pretrained EfficientNet model and replace its classifier
model = models.efficientnet_b7(weights='DEFAULT')
model.classifier[1] = AugmentHead(dim=model.classifier[1].in_features)  # Replace classifier head
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device
model = model.to(device)

# Load the initial model checkpoint
checkpoint_path = 'path/to/best_initial_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)

# Custom dataset for loading Dog Heart images and labels
class DogHeartDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # Load all image file paths and corresponding label paths
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]
        self.points = list(sorted(os.listdir(os.path.join(root, "Labels"))))

    def __getitem__(self, idx):
        # Load image and label for the given index
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.points[idx])
        img = Image.open(img_path).convert("RGB")  # Convert image to RGB format
        w, h = img.size
        if self.transforms is not None:
            img = self.transforms(img)  # Apply transformations
        h_new, w_new = img.shape[1], img.shape[2]
        mat = loadmat(points_path)  # Load label data from .mat file
        six_points = mat['six_points'].astype(float)
        six_points = torch.as_tensor(six_points, dtype=torch.float32)
        # Adjust points for resized image dimensions
        six_points[:, 0] = w_new / w * six_points[:, 0]
        six_points[:, 1] = h_new / h * six_points[:, 1]
        six_points = torch.reshape(six_points, (-1,)) / h_new  # Normalize points
        VHS = mat['VHS'].astype(float)
        VHS = torch.as_tensor(VHS, dtype=torch.float32).reshape([1, 1])  # Reshape VHS data
        return idx, img, six_points, VHS

    def __len__(self):
        return len(self.imgs)  # Return the total number of images

# Define image transformations
def get_transform(resized_image_size):
    transforms = []
    transforms.append(T.ToTensor())  # Convert PIL image to tensor
    transforms.append(T.Resize((resized_image_size, resized_image_size)))  # Resize image
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  # Normalize image
    return T.Compose(transforms)

# Initialize datasets and data loaders
resized_image_size = 512
true_batch_size = 16
accumulation_steps = 4

# Create training and validation datasets
dataset_train = DogHeartDataset('path/to/original_dataset/Train', get_transform(resized_image_size))
dataset_valid = DogHeartDataset('path/to/original_dataset/Valid', get_transform(resized_image_size))
train_loader = DataLoader(dataset_train, batch_size=true_batch_size // accumulation_steps, shuffle=True, num_workers=8) 
valid_loader = DataLoader(dataset_valid, batch_size=8, shuffle=False, num_workers=8)

# Custom test dataset for inference
class DogHeartTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")  # Convert image to RGB
        if self.transforms:
            img = self.transforms(img)  # Apply transformations
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)  # Total number of test images

# Load test and unlabeled datasets
test_dataset = DogHeartTestDataset('path/to/original_dataset/Test_Images', get_transform(resized_image_size))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

unlabeled_dataset = DogHeartTestDataset('path/to/generated_dataset', get_transform(resized_image_size))
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=true_batch_size // accumulation_steps, shuffle=False, num_workers=8)

# Function to calculate VHS from model outputs
def calc_vhs(x: torch.Tensor):
    A = x[..., 0:2]
    B = x[..., 2:4]
    C = x[..., 4:6]
    D = x[..., 6:8]
    E = x[..., 8:10]
    F = x[..., 10:12]
    # Calculate distances between key points
    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)
    # Compute VHS based on distances
    vhs = 6 * (AB + CD) / EF
    return vhs

# Custom dataset for high-confidence pseudo-labels
class HighConfidenceDataset(Dataset):
    def __init__(self, images, pseudo_labels):
        self.images = images
        self.pseudo_labels = pseudo_labels
    
    def __len__(self):
        return len(self.images)  # Return number of high-confidence samples
    
    def __getitem__(self, idx):
        img = self.images[idx]
        pseudo_label = self.pseudo_labels[idx]
        # Calculate VHS for pseudo-labels
        pseudo_vhs = calc_vhs(pseudo_label.unsqueeze(0)).reshape([1, 1])  # Reshape for compatibility
        idx_dummy = -1  # Dummy index to maintain compatibility with existing datasets
        return idx_dummy, img, pseudo_label, pseudo_vhs

# Define training hyperparameters
learning_rate = 1e-5
num_epochs = 100

# Define loss function, optimizer, and learning rate scheduler
criterion = torch.nn.L1Loss()  # Use L1 Loss for regression tasks
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)  # AdamW optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # Cosine Annealing Scheduler

# Base path for saving outputs
base_path = "path/to/save/outputs"
for folder in ['predictions', 'models']:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)  # Create directories for saving predictions and models

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    optimizer.zero_grad()  # Clear gradients before training

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
        ind, images, points, vhs = data
        images = images.to(device)
        points = points.to(device)
        vhs = vhs.to(device)
        labels = ((vhs >= 10).long() - (vhs < 8.2).long() + 1).squeeze()  # Generate binary labels based on VHS
        outputs = model(images)  # Forward pass
        loss1 = criterion(outputs.squeeze(), points.squeeze())  # Point regression loss
        loss2 = criterion(calc_vhs(outputs).squeeze(), vhs.squeeze())  # VHS regression loss
        loss = 10 * loss1 + 0.1 * loss2  # Combine losses with weights

        # Record individual losses for monitoring
        running_loss1 += loss1.item() * images.size(0)
        running_loss2 += loss2.item() * images.size(0)
        loss = loss / accumulation_steps  # Scale loss for gradient accumulation
        loss.backward()  # Backpropagation

        # Perform optimizer step after accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)  # Calculate average training loss

    scheduler.step()  # Update learning rate scheduler

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_loss1 = 0.0
    val_loss2 = 0.0
    val_correct = 0
    with torch.no_grad():
        for _, images, points, vhs_gt in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            points = points.to(device)
            vhs_gt = vhs_gt.to(device)
            labels = ((vhs_gt >= 10).long() - (vhs_gt < 8.2).long() + 1).squeeze()
            outputs = model(images)  # Forward pass
            vhs_pred = calc_vhs(outputs)  # Calculate VHS predictions
            label_pred = ((vhs_pred >= 10).long() - (vhs_pred < 8.2).long() + 1).squeeze()
            loss1 = criterion(outputs, points)  # Point loss
            loss2 = criterion(vhs_pred.squeeze(), vhs_gt.squeeze())  # VHS loss
            loss = 10 * loss1 + 0.1 * loss2
            val_loss += loss.item() * images.size(0)
            val_loss1 += loss1.item() * images.size(0)
            val_loss2 += loss2.item() * images.size(0)
            val_correct += label_pred.eq(labels).sum().item()  # Calculate accuracy
    val_loss = val_loss / len(valid_loader.dataset)  # Average validation loss
    val_acc = val_correct / len(valid_loader.dataset)  # Validation accuracy

    # Print metrics
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss}, Valid Loss: {val_loss}, Valid Acc: {val_acc}')

    # Save metrics to CSV
    df_loss = pd.DataFrame([{
        'Epoch': epoch + 1,
        'Train': epoch_loss,
        'Train_Loss1': running_loss1 / len(train_loader.dataset),
        'Train_Loss2': running_loss2 / len(train_loader.dataset),
        'Valid': val_loss,
        'Valid_Loss1': val_loss1 / len(valid_loader.dataset),
        'Valid_Loss2': val_loss2 / len(valid_loader.dataset),
        'Acc': val_acc
    }])
    file_path = os.path.join(base_path, 'loss.csv')
    df_loss.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

    # Test loop: Generate predictions
    img_names = []
    vhs_pred_list = []
    with torch.no_grad():
        for inputs, img_name in test_loader:
            inputs = inputs.to(device)
            img_names += list(img_name)
            outputs = model(inputs)  # Forward pass
            vhs_pred = calc_vhs(outputs)  # Calculate VHS predictions
            vhs_pred_list += list(vhs_pred.detach().cpu().numpy())

        # Save predictions to CSV
        df = pd.DataFrame({'ImageName': img_names, 'VHS': vhs_pred_list})
        file_path = os.path.join(base_path, "predictions", f"predictions_epoch_{epoch+1}.csv")
        df.to_csv(file_path, index=False, header=False)

    # Save the best model based on validation accuracy
    if val_acc >= acc:
        best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        file_path = os.path.join(base_path, "models", f"best_model_epoch_{epoch+1}.pth")
        torch.save(best_model, file_path)
        print('Model saved!')

    # Generate pseudo-labels using MC Dropout for unlabeled data
    model.train()  # Enable dropout during inference for MC Dropout
    pseudo_labels = []
    uncertainties = []
    with torch.no_grad():
        for images, img_name in tqdm(unlabeled_loader, desc="Generating Pseudo-labels with Uncertainty"):
            images = images.to(device)
            preds = []
            # Perform multiple stochastic forward passes
            for _ in range(20):  # Number of MC Dropout samples
                preds.append(model(images).cpu())
            preds = torch.stack(preds)
            mean_preds = preds.mean(dim=0)  # Mean prediction
            std_preds = preds.std(dim=0)    # Standard deviation as uncertainty

            pseudo_labels.append(mean_preds)
            uncertainties.append(std_preds)

    pseudo_labels = torch.cat(pseudo_labels)  # Combine all pseudo-labels
    uncertainties = torch.cat(uncertainties)  # Combine all uncertainties

    # Filter high-confidence pseudo-labels based on uncertainty threshold
    confidence_threshold = 0.005
    high_confidence_indices = torch.where(uncertainties.max(dim=1)[0] < confidence_threshold)[0]

    if len(high_confidence_indices) > 0:
        # Collect high-confidence images and pseudo-labels
        x_high_confidence = torch.stack([
            unlabeled_loader.dataset[idx][0] for idx in high_confidence_indices
        ])
        pseudo_labels_high_confidence = pseudo_labels[high_confidence_indices].detach().clone()
        print(f"High confidence pseudo-labels count: {len(high_confidence_indices)} (out of {len(unlabeled_loader.dataset)})")

        # Create a high-confidence dataset
        high_confidence_dataset = HighConfidenceDataset(
            x_high_confidence,
            pseudo_labels_high_confidence
        )
        # Concatenate with the original training dataset
        combined_dataset = ConcatDataset([dataset_train, high_confidence_dataset])
        # Update the training data loader
        train_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=8)
    else:
        print("No high confidence pseudo-labels found.")  # No changes to training dataset

# Save the final model after training
last_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
torch.save(last_model, os.path.join(base_path, "models", "last_model.pth"))
