import torch
import torch.nn as nn
import torchvision.models as models

# Define a custom augmentation head
class AugmentHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.augment_head = nn.Linear(dim, 12)  # Output size is 12

    def forward(self, x):
        return self.augment_head(x)

# Load EfficientNet model and modify its classifier
def get_model(device):
    model = models.efficientnet_b7(weights='DEFAULT')
    model.classifier[1] = AugmentHead(dim=model.classifier[1].in_features)
    return model.to(device)

# Function to calculate VHS from model outputs
def calc_vhs(x: torch.Tensor):
    A = x[..., 0:2]
    B = x[..., 2:4]
    C = x[..., 4:6]
    D = x[..., 6:8]
    E = x[..., 8:10]
    F = x[..., 10:12]
    
    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)
    
    vhs = 6 * (AB + CD) / EF
    
    return vhs
