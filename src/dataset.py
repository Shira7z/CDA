import os
from PIL import Image
import torch
from scipy.io import loadmat
import torchvision.transforms as T
from torch.utils.data import Dataset

# Custom dataset for labeled data
class DogHeartDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(os.path.join(root, "Images")) if f.endswith(('png', 'jpg'))])
        self.points = sorted(os.listdir(os.path.join(root, "Labels")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.points[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if self.transforms:
            img = self.transforms(img)
        mat = loadmat(points_path)
        six_points = torch.tensor(mat['six_points'], dtype=torch.float32)
        six_points[:,0] = six_points[:,0] / w
        six_points[:,1] = six_points[:,1] / h
        VHS = torch.tensor(mat['VHS'], dtype=torch.float32)
        return idx, img, six_points, VHS

    def __len__(self):
        return len(self.imgs)

# Dataset for test and unlabeled data
class DogHeartTestDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(os.path.join(root, "Images")) if f.endswith(('png', 'jpg'))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

# Dataset for High Confidence Data
class HighConfidenceDataset(Dataset):
    def __init__(self, images, pseudo_labels):
        self.images = images
        self.pseudo_labels = pseudo_labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        pseudo_label = self.pseudo_labels[idx]
        pseudo_vhs = calc_vhs(pseudo_label.unsqueeze(0)).reshape([1,1]) # Add batch dimension for calc_vhs compatibility
        idx_dummy = -1  # This is not used but added to keep compatibility
        return idx_dummy, img, pseudo_label, pseudo_vhs
    

