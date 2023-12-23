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


def get_loss(model, dataloader, criterion, pred_fault_loc=True, device='cpu'):
    model.eval()
    running_loss = 0
    running_examples = 0

    with torch.no_grad():
        for data in dataloader:
            X = data["data"].type(torch.float32)
            y = data["label"].type(torch.float32)
            y = format_target(y, pred_fault_loc=pred_fault_loc)
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            running_loss += loss
            running_examples += len(X)

    loss = np.round((running_loss/running_examples).detach().cpu().numpy(), 7)
    return loss


def get_accuracy(model, dataloader, pred_fault_loc=True, device='cpu',):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            X = data["data"].type(torch.float32)
            y = data["label"].type(torch.float32)
            y = format_target(y, pred_fault_loc=pred_fault_loc)
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = torch.where(y_pred > 0.5, 1, 0)

            correct += sum(y == y_pred)
            total += len(y)

    accuracy = np.round((correct/total).detach().cpu().numpy(), 3)
    return accuracy
