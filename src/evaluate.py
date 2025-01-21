import torch
from tqdm import tqdm
import pandas as pd
from model import calc_vhs
import matplotlib.pyplot as plt

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


# Function to plot images with their predicted and true points (if applicable).
def plot_predictions(model, data_loader, device):
    model.eval()  
    with torch.no_grad():
        for data_batch in data_loader:
            if len(data_batch) == 4:
                _, images, points, vhs_gt = data_batch
                batch_size = images.size(0)

                images = images.to(device)
                points = points.to(device)
                vhs_gt = vhs_gt.to(device)
                outputs = model(images)
                vhs_pred = calc_vhs(outputs)                                  

                images = images.permute(0, 2, 3, 1).cpu().numpy()
                points = points.cpu().numpy()
                outputs = outputs.cpu().numpy()                                                    
                vhs_gt = vhs_gt.cpu().numpy()
                vhs_pred = vhs_pred.cpu().numpy()

                fig, axes = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
                if batch_size == 1:
                    axes = [axes]   

                for i, ax in enumerate(axes):
                    img = images[i]
                    img = (img - img.min()) / (img.max() - img.min())
                    true_points = points[i].reshape(-1, 2) * img.shape[0]
                    pred_points = outputs[i].reshape(-1, 2) * img.shape[0]
                    true_vhs = vhs_gt[i].item()
                    pred_vhs = vhs_pred[i].item()
                    
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.scatter(true_points[:, 0], true_points[:, 1], c='blue', label='True Points')
                    ax.plot(true_points[[0, 1], 0], true_points[[0, 1], 1], 'blue')
                    ax.plot(true_points[[2, 3], 0], true_points[[2, 3], 1], 'blue')
                    ax.plot(true_points[[4, 5], 0], true_points[[4, 5], 1], 'blue')
                    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='red', label='Prediction')
                    ax.plot(pred_points[[0, 1], 0], pred_points[[0, 1], 1], 'red')
                    ax.plot(pred_points[[2, 3], 0], pred_points[[2, 3], 1], 'red')
                    ax.plot(pred_points[[4, 5], 0], pred_points[[4, 5], 1], 'red')
                    ax.set_title(f"True: {true_vhs:.2f}, Pred: {pred_vhs:.2f}")
                    ax.legend()
              
                plt.tight_layout()
                plt.show()


            if len(data_batch) == 2:
                images, _ = data_batch
                batch_size = images.size(0)

                images = images.to(device)
                outputs = model(images)
                vhs_pred = calc_vhs(outputs)                                  

                images = images.permute(0, 2, 3, 1).cpu().numpy()
                outputs = outputs.cpu().numpy()   
                vhs_pred = vhs_pred.cpu().numpy()

                fig, axes = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
                if batch_size == 1:
                    axes = [axes]   

                for i, ax in enumerate(axes):
                    img = images[i]
                    img = (img - img.min()) / (img.max() - img.min())
                    pred_points = outputs[i].reshape(-1, 2) * img.shape[0]
                    pred_vhs = vhs_pred[i].item()

                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='red', label='Prediction')
                    ax.plot(pred_points[[0, 1], 0], pred_points[[0, 1], 1], 'red')
                    ax.plot(pred_points[[2, 3], 0], pred_points[[2, 3], 1], 'red')
                    ax.plot(pred_points[[4, 5], 0], pred_points[[4, 5], 1], 'red')
                    ax.set_title(f"Predicted VHS: {pred_vhs:.2f}")
                    ax.legend()
              
                plt.tight_layout()
                plt.show()
