import os
from pathlib import Path
import torch
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import DigitalTwinDataset
from utils import format_target
from evaluate import get_accuracy, get_loss, predict_targets
from models import load_model
from engine import train_model

# Define training device and set seed for re-implementation
torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
    print("CUDA is available.")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    torch.mps.manual_seed(42)
    print("MPS is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Define dataset hyperparameters
sample_length = 512
shuffle = True
batch_size = 32

# Create datasets and dataloaders
train_dataset = DigitalTwinDataset(
    "datasets/digital_twins/train/", sample_length, device, shuffle)
valid_dataset, test_dataset = DigitalTwinDataset(
    "datasets/digital_twins/test/", sample_length, device, shuffle
).split(0.5)

# Create dataloaders for the three datsets
train_data_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle)
valid_data_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=shuffle)
test_data_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=shuffle)

# Instantiate a model
pred_fault_loc = True
mask_pred_loc = True
model = load_model(d_input_channels=6,
                   pred_fault_loc=pred_fault_loc,
                   mask_pred_loc=mask_pred_loc,
                   device=device
                   )
model.to(device)

# Define an optimizer for the model
learning_rate = .005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define a scheduler for the optimizer
n_epochs = 200
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=n_epochs//4, gamma=0.5)

# Define a Loss funtion
criterion = torch.nn.BCELoss()


# Train the model
if __name__ == '__main__':
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
        model=model,
        train_dataloader=train_data_loader,
        valid_dataloader=valid_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        pred_fault_loc=pred_fault_loc,
        n_epochs=n_epochs,
        device=device
    )

    # Save model outputs
    ct = datetime.datetime.now()
    log_path = os.path.join(
        'checkpoints', f"{ct.year}-{ct.month}-{ct.day}/{ct.hour}-{ct.minute}"
    )
    if not os.path.exists(log_path):
        Path(log_path).mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_losses, columns=['Loss'])
    train_df[['Fault Class', 'Fault Location 1', 'Fault Location 2',
              'Fault Location 3']] = pd.DataFrame(train_accuracies)

    valid_df = pd.DataFrame(valid_losses, columns=['Loss'])
    valid_df[['Fault Class', 'Fault Location 1', 'Fault Location 2',
              'Fault Location 3']] = pd.DataFrame(valid_accuracies)

    torch.save(model.state_dict(), os.path.join(log_path, 'model_weights.pth'))
    train_df.to_csv(os.path.join(log_path, "train_results.csv"), index=False)
    valid_df.to_csv(os.path.join(log_path, "valid_results.csv"), index=False)
