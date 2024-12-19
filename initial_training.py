import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.io import loadmat, savemat
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a custom augmentation head as a fully connected layer
class AugmentHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.augment_head = nn.Linear(dim, 12)  # Output size is 12

    def forward(self, x):
        x = self.augment_head(x)  # Forward pass through the augmentation head
        return x
      
# Load EfficientNet model and replace classifier head
model = models.efficientnet_b7(weights='DEFAULT')
model.classifier[1] = AugmentHead(dim=model.classifier[1].in_features)  # Modify classifier with AugmentHead
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability
model = model.to(device)

# Define a custom dataset class for Dog Heart data
class DogHeartDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # Load all image filenames and corresponding label filenames
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]
        self.points = list(sorted(os.listdir(os.path.join(root, "Labels"))))

    def __getitem__(self, idx):
        # Load image and corresponding label
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.points[idx])
        img = Image.open(img_path).convert("RGB")  # Convert image to RGB
        w, h = img.size
        if self.transforms is not None:
            img = self.transforms(img)  # Apply transformations
        h_new, w_new = img.shape[1], img.shape[2]
        mat = loadmat(points_path)  # Load label data from .mat file
        six_points = mat['six_points'].astype(float)
        six_points = torch.as_tensor(six_points, dtype=torch.float32)
        six_points[:, 0] = w_new / w * six_points[:, 0]  # Resize and maintain aspect ratio for x-coordinates
        six_points[:, 1] = h_new / h * six_points[:, 1]  # Resize and maintain aspect ratio for y-coordinates
        six_points = torch.reshape(six_points, (-1,)) / h_new  # Normalize the points
        VHS = mat['VHS'].astype(float)
        VHS = torch.as_tensor(VHS, dtype=torch.float32)
        return idx, img, six_points, VHS

    def __len__(self):
        return len(self.imgs)  # Return the size of the dataset

# Define transformation function for data preprocessing
def get_transform(resized_image_size):
    transforms = []
    transforms.append(T.ToTensor())  # Convert PIL image to tensor
    transforms.append(T.Resize((resized_image_size, resized_image_size)))  # Resize image
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  # Normalize image
    return T.Compose(transforms)

# Set image size, batch size, and accumulation steps
resized_image_size = 512
true_batch_size = 16
accumulation_steps = 8

# Create train and validation datasets and data loaders
dataset_train = DogHeartDataset('path/to/augmentation_dataset/Train', get_transform(resized_image_size))
dataset_valid = DogHeartDataset('path/to/origianl_dataset/Valid', get_transform(resized_image_size))
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=true_batch_size // accumulation_steps, shuffle=True) 
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=8, shuffle=False)

# Define a test dataset for inference
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
        return len(self.imgs)

# Instantiate test dataset and data loader
test_dataset = DogHeartTestDataset('path/to/original_dataset/Test_Images', get_transform(resized_image_size))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define a function to calculate VHS from model outputs
def calc_vhs(x: torch.Tensor):
    A = x[..., 0:2]
    B = x[..., 2:4]
    C = x[..., 4:6]
    D = x[..., 6:8]
    E = x[..., 8:10]
    F = x[..., 10:12]
    AB = torch.norm(A - B, p=2, dim=-1)  # Compute Euclidean distance between points A and B
    CD = torch.norm(C - D, p=2, dim=-1)  # Compute distance between points C and D
    EF = torch.norm(E - F, p=2, dim=-1)  # Compute distance between points E and F
    vhs = 6 * (AB + CD) / EF  # Calculate VHS based on distances
    return vhs

# Training hyperparameters
learning_rate = 3e-4
num_epochs = 1000

# Define loss function, optimizer, and scheduler
criterion = torch.nn.L1Loss()  # Use L1 loss for regression
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Create directories for saving predictions and models
for folder in ['predictions', 'models']:
    os.makedirs(folder, exist_ok=True)

# Initialize variables to track training and validation metrics
train_loss = []
valid_loss = []
valid_acc = []
pred_record = torch.zeros([len(train_loader.dataset), 10, 12])

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    optimizer.zero_grad()  # Zero the gradients before each epoch
    for batch_idx, (ind, images, points, vhs) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
        images = images.to(device)
        points = points.to(device)
        vhs = vhs.to(device)
        labels = ((vhs >= 10).long() - (vhs < 8.2).long() + 1).squeeze()  # Generate classification labels
        outputs = model(images)  # Forward pass
        loss1 = criterion(outputs.squeeze(), points.squeeze())  # Loss for regression points
        loss2 = criterion(calc_vhs(outputs).squeeze(), vhs.squeeze())  # Loss for VHS calculation
        loss = 10 * loss1 + 0.1 * loss2  # Combine losses with weights
        if epoch > 10:
            soft_points = pred_record[ind].mean(axis=1).to(device)  # Use soft labels from previous predictions
            loss3 = criterion(outputs.squeeze(), soft_points)  # Soft label loss
            loss += loss3
        loss = loss / accumulation_steps  # Scale loss for accumulation
        loss.backward()  # Backpropagation
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Zero gradients
        running_loss += loss.item() * accumulation_steps * images.size(0)
        pred_record[ind, epoch % 10] = outputs.detach().cpu()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)  # Track training loss
    scheduler.step()  # Update learning rate scheduler

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for _, images, points, vhs_gt in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            points = points.to(device)
            vhs_gt = vhs_gt.to(device)
            labels = ((vhs_gt >= 10).long() - (vhs_gt < 8.2).long() + 1).squeeze()
            outputs = model(images)
            vhs_pred = calc_vhs(outputs)  # Calculate VHS predictions
            label_pred = ((vhs_pred >= 10).long() - (vhs_pred < 8.2).long() + 1).squeeze()
            loss1 = criterion(outputs, points)  # Validation point loss
            loss2 = criterion(vhs_pred.squeeze(), vhs_gt.squeeze())  # Validation VHS loss
            loss = 10 * loss1 + 0.1 * loss2
            val_loss += loss.item() * images.size(0)  # Accumulate validation loss
            val_correct += label_pred.eq(labels).sum().item()  # Count correct predictions
    val_loss = val_loss / len(valid_loader.dataset)
    valid_loss.append(val_loss)  # Track validation loss
    val_acc = val_correct / len(valid_loader.dataset)  # Calculate validation accuracy
    valid_acc.append(val_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss}, Valid Loss: {val_loss}, Valid Acc: {val_acc}')
    
    # Save training metrics to CSV
    df_loss = pd.DataFrame({'Train': train_loss, 'Valid': valid_loss, 'Acc': valid_acc})
    df_loss.to_csv(f'loss.csv', index=False, header=True)
    
    # Inference on test dataset
    img_names = []
    vhs_pred_list = []
    with torch.no_grad():
        for inputs, img_name in test_loader:
            inputs = inputs.to(device)
            img_names += list(img_name)
            outputs = model(inputs)
            vhs_pred = calc_vhs(outputs)
            vhs_pred_list += list(vhs_pred.detach().cpu().numpy())

        df = pd.DataFrame({'ImageName': img_names, 'VHS': vhs_pred_list})
        df.to_csv(f'predictions/predictions_epoch_{epoch+1}.csv', index=False, header=False)
    
    # Save best model based on validation accuracy
    if val_acc > acc:
        best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_model, f'models/best_model_epoch_{epoch+1}.pth')
        print('Model saved!')        
    
# Save the final model
last_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
torch.save(last_model, "models/last_model.pth")
