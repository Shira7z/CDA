import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import DogHeartDataset
from model import get_model, calc_vhs

def train_initial_model(train_loader, valid_loader, device, num_epochs=1000, lr=3e-4):
    model = get_model(device)
    criterion = torch.nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    pred_record = torch.zeros([len(train_loader.dataset), 10, 12])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (ind, images, points, vhs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, points, vhs = images.to(device), points.to(device), vhs.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss1 = criterion(outputs, points)
            loss2 = criterion(calc_vhs(outputs), vhs)
            loss = 10 * loss1 + 0.1 * loss2
            if epoch > 10:
                soft_points = pred_record[ind].mean(axis=1).to(device)
                loss3 = criterion(outputs, soft_points)
                loss += loss3
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_record[ind, epoch % 10] = outputs.detach().cpu()

        scheduler.step()
