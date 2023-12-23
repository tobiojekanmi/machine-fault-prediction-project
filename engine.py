import torch
import numpy as np
from utils import format_target
from evaluate import get_accuracy, get_loss


def train_model(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        scheduler=None,
        pred_fault_loc=True,
        n_epochs=100,
        device='cpu'
):
    """
    Trainer function.

    Args:
        model: model to train
        train_dataloader: Train DataLoader
        valid_dataloader: Validation DataLoader
        criterion: Loss or objective function to minimize
        optimizer: Optimizer function
        scheduler (optional): Scheduler function if desired. Defaults to None.
        pred_fault_loc (bool, optional): Model predicts fault locations or not. Defaults to True.
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        device (str, optional): Training and validation device. Defaults to 'cpu'.

    """
    model.train()
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []

    for epoch in range(n_epochs):
        iter = 0
        running_loss = 0.
        running_examples = 0
        for i, batch in enumerate(train_dataloader):
            # Split batch to inputs and targets
            X = batch["data"].type(torch.float32)
            y = batch["label"].type(torch.float32)
            y = format_target(y, pred_fault_loc=pred_fault_loc)
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X).type(torch.float32)

            # Calculate loss
            loss = criterion(y_pred, y)
            running_loss += loss
            running_examples += len(X)

            # Zero the Optimizer gradients
            optimizer.zero_grad()

            # Compute parameters gradients using the Loss backward
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Update learning rate using scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate model performance using train and validation data
        train_accuracy = get_accuracy(
            model, train_dataloader, pred_fault_loc, device)
        valid_accuracy = get_accuracy(
            model, valid_dataloader, pred_fault_loc, device)

        # Get train and validation loss
        train_loss = get_loss(model, train_dataloader,
                              criterion, pred_fault_loc, device)
        valid_loss = get_loss(model, valid_dataloader,
                              criterion, pred_fault_loc, device)

        # Save Losses and accs
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)

        # Display loss after every epoch
        print(
            f'Epoch: {epoch+1}, lr: {optimizer.param_groups[0]["lr"] :.2}, \
                Train loss: {running_loss/running_examples:.2}, Train Acc: {train_accuracy}, \
                    Validation Acc: {valid_accuracy}'
        )

        iter += 1

    return np.array(train_losses), np.array(valid_losses), np.array(train_accs), np.array(valid_accs)
