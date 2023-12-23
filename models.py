import torch
from torch import nn
from typing import Tuple, Optional
from typing import Optional, Tuple


class CNN(nn.Module):
    def __init__(
        self,
        d_input: int = 512,
        d_input_channels: int = 6,
        n_conv_layers: int = 6,
        d_cnn_filter: int = 128,
        d_mlp_filter: int = 256,
        skip_layers: Tuple[int] = (2, 4),
        class_pred_threshold: float = 0.5,
        pred_fault_loc: Optional[bool] = False,
        mask_pred_loc: Optional[bool] = False,
    ):
        """        
        Convolutional Neural Network (CNN) for signal classification and, optionally,
        fault location prediction.
        Utilizes 1D convolution layers for feature extraction and linear layers for classification.

            Args:
                d_input (int): Dimension of the input signal.
                d_input_channels (int): Number of input channels in the signal.
                n_conv_layers (int): Number of convolutional layers in the network.
                d_cnn_filter (int): Number of filters in the convolutional layers.
                d_mlp_filter (int): Number of units in the multi-layer perceptron (MLP) layers.
                skip_layers (Tuple[int]): Index positions to skip concatenation in conv layers.
                class_pred_threshold (float): Threshold for classifying signals as healthy/faulty.
                pred_fault_loc (Optional[bool]): Enable/disable fault location prediction.
                mask_pred_loc (Optional[bool]): Enable/disable masking in fault location prediction.
        """

        super().__init__()
        self.d_input_channels = d_input_channels
        self.skip = skip_layers
        self.activation = nn.functional.relu
        self.pred_fault_loc = pred_fault_loc
        self.class_pred_threshold = class_pred_threshold
        self.mask_pred_loc = mask_pred_loc

        # Create model hidden layers using 1D convolutional layers
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(self.d_input_channels, d_cnn_filter,
                          kernel_size=5, padding='same')
            ] +
            [
                nn.Conv1d(d_cnn_filter + self.d_input_channels, d_cnn_filter, kernel_size=5,
                          padding='same') if i in skip_layers else
                nn.Conv1d(d_cnn_filter, d_cnn_filter,
                          kernel_size=5, padding='same')
                for i in range(n_conv_layers - 1)
            ]
        )

        # Create model output layer for binary healthy/faulty classification
        self.pool = nn.AvgPool1d(7, stride=2, padding=3)
        self.class_branch = nn.Linear(d_input//2 * d_cnn_filter, d_mlp_filter)
        self.class_out = nn.Linear(d_mlp_filter, 1)

        # Add extra layers to predict fault locations if enabled
        if self.pred_fault_loc:
            self.input_branch = nn.Linear(d_mlp_filter + 1, d_mlp_filter)
            self.branch = nn.Linear(d_mlp_filter, d_mlp_filter)
            self.loc_out = nn.Linear(d_mlp_filter, 3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        # Change input shape to (batch, 6, 512)
        x = x.permute(0, 2, 1)

        # Copy x for skip connection
        x_input = x.clone().detach().requires_grad_(True)

        # Implement the feature extraction with CNN layers
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            # Concatenate input with prev layer output if layer index is in skip
            if i in self.skip:
                x = torch.cat([x, x_input], dim=1)

        # Pool, flatten, and classify signal into healthy/faulty signal
        x = self.pool(x)
        x = x.flatten(start_dim=1).type(torch.float32)
        x = self.activation(self.class_branch(x))
        class_out = torch.sigmoid(self.class_out(x))

        # Predict fault locations if enabled
        if self.pred_fault_loc:
            # Concatenate signal class prediction with the last layer feature vector
            x = torch.cat([x, class_out], dim=1)

            # Implement a forward pass on the extra layers
            x = self.activation(self.input_branch(x))
            x = self.activation(self.branch(x))
            loc_out = torch.sigmoid(self.loc_out(x))

            # Mask the model output such that the fault location prediction is dependent on the
            # class output (i.e., set fault locations to 0 where class prediction is below
            # classification threshold)
            if self.mask_pred_loc:
                mask = class_out > self.class_pred_threshold
                loc_out = mask.expand(loc_out.shape) * loc_out

            # Concatenate and return class prediction and fault locations
            return torch.cat([class_out, loc_out], dim=1)
        else:
            # If not predicting fault locations, output only class prediction
            return class_out


def load_model(
        d_input: int = 512,
        d_input_channels: int = 6,
        n_conv_layers: int = 6,
        d_cnn_filter: int = 128,
        d_mlp_filter: int = 256,
        skip_layers: Tuple[int] = (2, 4),
        class_pred_threshold: float = 0.5,
        pred_fault_loc: Optional[bool] = False,
        mask_pred_loc: Optional[bool] = False,
        weights_path=None,
        device=None
):
    """
    Loads new or pre-trained model
    """

    model = CNN(
        d_input,
        d_input_channels,
        n_conv_layers,
        d_cnn_filter,
        d_mlp_filter,
        skip_layers,
        class_pred_threshold,
        pred_fault_loc,
        mask_pred_loc
    )

    if weights_path is not None:
        if device is not None:
            device = torch.device(device) if isinstance(
                device, str) else device
            model.load_state_dict(torch.load(
                weights_path, map_location=device))
        else:
            model.load_state_dict(torch.load(weights_path))

    return model
