"""Object model for SolDet."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetzLoss(torch.nn.Module):
    """1D loss function for soliton detection.

    Implementation of the loss function defined in: https://arxiv.org/abs/2111.04881
    The first term is essentially the weighted cross entropy probability for the cell belonging to the
    'soliton present' class.
    The second term is a mean-squared error for the fractional position within the cell.

    Parameters
    ----------
    param : float
        The weight value used when calculating the loss.
        (default = 5.7)

    """

    def __init__(self, param: float = 5.7) -> None:
        """Initialize loss function.

        Parameters
        ----------
        param : float
            The weight value used when calculating the loss.
            (default = 5.7)

        """
        super().__init__()
        self.param = param

    def forward(self, pred: torch.Tensor, tars: torch.Tensor) -> torch.Tensor:
        """Do one forward pass and calculate a loss.

        Parameters
        ----------
        pred : Tensor
            The output tensor of the neural network.
        tars : Tensor
            The target label tensor.

        Returns
        -------
        loss : Tensor
            The resulting loss value.

        """
        eps = 1e-10
        loss = torch.sum(-self.param * tars[:, 0, :, :] * torch.log(pred[:, 0, :, :] + eps)
                         - (1 - tars[:, 0, :, :]) * torch.log(1 - pred[:, 0, :, :] + eps)
                         + self.param * tars[:, 0, :, :] * (tars[:, 1, :, :] - pred[:, 1, :, :])**2)

        return loss / pred.shape[0]


class ObjectDetector(nn.Module):
    """Modern SolDet module Object Detector.

    This pytorch object detector model identifies the position of a solitonic excitation.

    Parameters
    ----------
    label_shape : tuple
        The shape of the position labels. For 1D this shape is typically (1, width // 4), where 4 is the number of
        pixels for each cell in the array.
        (default = 1, 41)

    """

    def __init__(self, label_shape: tuple[int, int] = (1, 41)) -> None:
        """Initialize the model.

        Parameters
        ----------
        label_shape : tuple
            The shape of the position labels. For 1D this shape is typically (1, width // 4), where 4 is the number of
            pixels for each cell in the array.
            (default = 1, 41)

        """
        super().__init__()

        n_layers = 4
        filter_list = [8, 16, 32, 64, 128]
        self.stack_size = 2
        kernel_size = 7
        in_channels = 1
        block_in_channels = 1
        self.p = 0.5

        self.dpth_conv_l = nn.ModuleList()
        self.pt_conv_l = nn.ModuleList()
        self.batch_norm_l = nn.ModuleList()
        self.res_conv_l = nn.ModuleList()
        for i, f in enumerate(filter_list[:n_layers]):
            self.dpth_conv_l.append(nn.ModuleList())
            self.pt_conv_l.append(nn.ModuleList())
            self.batch_norm_l.append(nn.ModuleList())

            for j in range(self.stack_size):
                if i < 3:
                    self.dpth_conv_l[i].append(
                    nn.Conv2d(
                    block_in_channels, f, (kernel_size, kernel_size), padding="same", groups=block_in_channels))
                else:
                    self.dpth_conv_l[i].append(
                    nn.Conv2d(block_in_channels, f, (1, kernel_size), padding="same", groups=block_in_channels))
                self.pt_conv_l[i].append(nn.Conv2d(f, f, 1))
                self.batch_norm_l[i].append(nn.BatchNorm2d(f))

                if j == self.stack_size - 1:
                    self.res_conv_l.append(nn.Conv2d(in_channels, f, 1))

                block_in_channels = f
            in_channels = f

        self.adaptive_pool = nn.AdaptiveAvgPool2d(label_shape)
        self.pconv = nn.Conv2d(in_channels, 128, (1, kernel_size), padding="same")
        self.fconv = nn.Conv2d(128, 2, (1, kernel_size), padding="same")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a tensor and identify any solitons and their positions.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, 2, 1, W // 4)
            The output tensor containing the probabilities for a soliton to be present in one of the 41 cells and its
            fractional position within a cell. Here B is the batch size and dimension 1 contains the probability (0)
            and the position within the cell (1), where values 0 to 1 indicate left to right. Dimension 2 indicates the
            number of cells in the vertical direction and dimension 3 indicates the number of cells in the horizontal
            direction.

        """
        for i in range(len(self.dpth_conv_l)):
            res = x
            for j in range(self.stack_size):
                x = functional.relu(functional.dropout(self.pt_conv_l[i][j](self.dpth_conv_l[i][j](x)), self.p))
                x = self.batch_norm_l[i][j](x)
                if j == self.stack_size - 1:

                    if i < 2:
                        x = functional.max_pool2d(x, (4, 2))
                        x += self.res_conv_l[i](functional.max_pool2d(res, (4, 2), (4, 2)))
                    elif i == 2:
                        x = functional.max_pool2d(x, (4, 1))
                        x += self.res_conv_l[i](functional.max_pool2d(res, (4, 1), (4, 1)))
                    elif i > 2:
                        x = self.adaptive_pool(x)
                        x += self.res_conv_l[i](self.adaptive_pool(res))

        x = functional.relu(functional.dropout(self.pconv(x), self.p))

        return torch.special.expit(self.fconv(x))
