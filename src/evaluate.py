import torch
from tqdm import tqdm
import pandas as pd
from model import calc_vhs
from dataset import DogHeartTestDataset
from torch.utils.data import DataLoader

# Function to evaluate the model on a given dataset
def evaluate_model(model, data_loader, device, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for _, images, points, vhs_gt in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            points = points.to(device)
            vhs_gt = vhs_gt.to(device)

            # Ground truth labels for classification
            labels = ((vhs_gt >= 10).long() - (vhs_gt < 8.2).long() + 1).squeeze()
            outputs = model(images)
            vhs_pred = calc_vhs(outputs)  # Calculate VHS predictions

            # Predicted classification labels
            label_pred = ((vhs_pred >= 10).long() - (vhs_pred < 8.2).long() + 1).squeeze()

            # Compute losses
            loss1 = criterion(outputs, points)
            loss2 = criterion(vhs_pred.squeeze(), vhs_gt.squeeze())
            loss = 10 * loss1 + 0.1 * loss2
            val_loss += loss.item() * images.size(0)
            val_correct += label_pred.eq(labels).sum().item()

    val_loss /= len(data_loader.dataset)
    val_acc = val_correct / len(data_loader.dataset)
    return val_loss, val_acc


# Function to run inference on a test dataset and save predictions
def inference_and_save(model, test_loader, device, output_path):
    model.eval()
    img_names = []
    vhs_predictions = []

    with torch.no_grad():
        for images, img_names_batch in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            vhs_pred = calc_vhs(outputs)  # Calculate VHS predictions

            # Collect image names and predictions
            img_names.extend(img_names_batch)
            vhs_predictions.extend(vhs_pred.cpu().numpy())

    # Save predictions to CSV
    df = pd.DataFrame({'ImageName': img_names, 'VHS': vhs_predictions})
    df.to_csv(output_path, index=False, header=True)
    print(f"Predictions saved to {output_path}")
