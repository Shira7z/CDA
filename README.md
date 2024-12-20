# CDA
## Confident Pseudo-labeled Diffusion Augmentation for Canine Cardiomegaly Detection

This repository provides the code and framework for detecting canine cardiomegaly using deep learning techniques. The pipeline includes initial training, iterative training with pseudo-labeling, evaluation, and inference.

## Features

- **Initial Training**: Train the model on labeled data to obtain the best initial model.
- **Iterative Training**: Improve model performance using pseudo-labeling with unlabeled data.
- **Evaluation**: Evaluate the model's performance on validation or test datasets.
- **Inference**: Generate predictions for test datasets and save them for analysis.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shira7z/CDA.git
   cd CDA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
The repository is organized as follows:
```bash
src/
|
├── dataset.py          # Dataset loaders for labeled and unlabeled data
├── model.py            # Model definition and VHS calculation
├── initial_train.py    # Initial training script
├── iterative_train.py  # Iterative training with pseudo-labeling
├── pseudo_labeling.py  # High-confidence pseudo-label generation
├── evaluate.py         # Evaluation and inference functions
├── utils.py            # Helper functions
```

## Usage

1. Initial Training
Train the model with labeled data to obtain the best initial model:
```python
from dataset import DogHeartDataset
from model import get_model
from torch.utils.data import DataLoader
from initial_train import train_initial_model
from utils import get_transform

# Load datasets
train_dataset = DogHeartDataset('path/to/train_dataset', transforms=get_transform(512))
valid_dataset = DogHeartDataset('path/to/valid_dataset', transforms=get_transform(512))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_initial_model(train_loader, valid_loader, device, num_epochs=100, lr=3e-4)
```

2. Iterative Training with Pseudo-Labeling
Refine the model using unlabeled data:
```python
from dataset import DogHeartDataset, DogHeartTestDataset
from iterative_train import train_with_pseudo_labels

train_dataset = DogHeartDataset('path/to/train_dataset', transforms=get_transform(512))
unlabeled_dataset = DogHeartTestDataset('path/to/unlabeled_dataset', transforms=get_transform(512))
valid_dataset = DogHeartDataset('path/to/valid_dataset', transforms=get_transform(512))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, device, num_epochs=50, lr=1e-5)
```


