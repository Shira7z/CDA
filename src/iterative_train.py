import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from dataset import HighConfidenceDataset
from pseudo_labeling import generate_pseudo_labels
from model import calc_vhs

def train_with_pseudo_labels(
    train_loader, unlabeled_loader, device, model, optimizer, criterion, scheduler, num_epochs
):
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
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_record[ind, epoch % 10] = outputs.detach().cpu()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        scheduler.step()

        # Generate pseudo-labels
        pseudo_labels, high_conf_idx = generate_pseudo_labels(model, unlabeled_loader, device)
        if len(high_conf_idx) > 0:
            high_conf_images = [unlabeled_loader.dataset[i][0] for i in high_conf_idx]
            high_conf_dataset = HighConfidenceDataset(high_conf_images, pseudo_labels)
            combined_dataset = ConcatDataset([train_loader.dataset, high_conf_dataset])
            train_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True)
