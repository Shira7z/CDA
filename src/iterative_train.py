import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from dataset import HighConfidenceDataset
from pseudo_labeling import generate_pseudo_labels
from model import calc_vhs

def train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, checkpoint, device, num_epochs=100, lr=1e-5):
    model = get_model(device)
    model.load_state_dict(checkpoint)
    criterion = torch.nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
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
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, valid_loader, device, criterion)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Generate pseudo-labels
        high_conf_pseudo_labels, high_conf_idx = generate_pseudo_labels(model, unlabeled_loader, device)
        if len(high_conf_idx) > 0:
            high_conf_images = [unlabeled_loader.dataset[i][0] for i in high_conf_idx]
            print(f"High confidence pseudo-labels count: {len(high_conf_idx)} (out of {len(unlabeled_loader.dataset)})")
            high_conf_dataset = HighConfidenceDataset(high_conf_images, high_conf_pseudo_labels)
            combined_dataset = ConcatDataset([train_loader.dataset, high_conf_dataset])
            train_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True)
        else:
            print("No high confidence pseudo-labels found.")
