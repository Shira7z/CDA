import torch
from tqdm import tqdm

def generate_pseudo_labels(model, unlabeled_loader, device, mc_passes=20, threshold=0.005):
    model.train()
    pseudo_labels, uncertainties = [], []

    for images, _ in tqdm(unlabeled_loader, desc="Generating Pseudo-labels"):
        images = images.to(device)
        preds = []
        for _ in range(mc_passes):
            preds.append(model(images).detach().cpu())
        preds = torch.stack(preds)
        mean_preds = preds.mean(dim=0)
        std_preds = preds.std(dim=0)
        pseudo_labels.append(mean_preds)
        uncertainties.append(std_preds)

    pseudo_labels = torch.cat(pseudo_labels)
    uncertainties = torch.cat(uncertainties)
    high_conf_idx = (uncertainties.max(dim=1)[0] < threshold).nonzero(as_tuple=True)[0]
    return pseudo_labels[high_conf_idx], high_conf_idx
