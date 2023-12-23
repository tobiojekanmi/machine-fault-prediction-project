import torch
import numpy as np
from utils import format_target


def predict_targets(model, dataloader, pred_fault_loc=True, device='cpu'):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data in dataloader:
            X = data["data"].type(torch.float32)
            y = data["label"].type(torch.float32)
            y = format_target(y, pred_fault_loc=pred_fault_loc)
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = torch.where(y_pred > 0.5, 1, 0)
            predictions.append(y_pred)
            targets.append(y)

        predictions = torch.concat(predictions, dim=0).cpu().detach().numpy()
        targets = torch.concat(targets, dim=0).cpu().detach().numpy()

    return targets, predictions

