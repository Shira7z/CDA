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
        VHS = torch.tensor(mat['VHS'], dtype=torch.float32).reshape([1])
        return idx, img, six_points / h, VHS

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
