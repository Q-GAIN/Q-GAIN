"""Classifier model for SolDet."""
import torch
from torch import nn
from torch.nn import functional

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLST2021CNNmodern(nn.Module):
    """Modern SolDet classifier.

    This pytorch classifier model identifies the class a solitonic image belongs to.

    Parameters
    ----------
    num_classes : int
        The number of classes to identify. By default this value is three classes, which represents the presence of
        no excitations (0), a single excitation (1), or multiple excitations (2).
        (default = 3)

    """

    def __init__(self, num_classes: int = 3) -> None:
        """Initialize model.

        Parameters
        ----------
        num_classes : int
            The number of classes to identify. By default this value is three classes, which represents the presence of
            no excitations (0), a single excitation (1), or multiple excitations (2).
            (default = 3)

        """
        super().__init__()

        filter_list = [8, 16, 32, 64, 128]
        self.stack_size = 3
        kernel_size = 8
        in_channels = 1
        block_in_channels = 1

        self.dpth_conv_l = nn.ModuleList()
        self.pt_conv_l = nn.ModuleList()
        self.batch_norm_l = nn.ModuleList()
        self.res_conv_l = nn.ModuleList()
        for i, f in enumerate(filter_list):
            self.dpth_conv_l.append(nn.ModuleList())
            self.pt_conv_l.append(nn.ModuleList())
            self.batch_norm_l.append(nn.ModuleList())

            for j in range(self.stack_size):
                self.dpth_conv_l[i].append(
                    nn.Conv2d(
                        block_in_channels, f, kernel_size, padding="same",
                        groups=block_in_channels))
                self.pt_conv_l[i].append(nn.Conv2d(f, f, 1))
                self.batch_norm_l[i].append(nn.BatchNorm2d(f))

                if j == self.stack_size - 1:
                    self.res_conv_l.append(nn.Conv2d(in_channels, f, 1))

                block_in_channels = f
            in_channels = f

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a tensor and make a class prediction.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, 3)
            The output tensor containing the probabilities for each class. The output is of shape (B, 3) where B is the
            batch size and the last dimension contains the probabilities for the number of classes specified in the
            model initialization.

        """
        # Max pooling over 2,2, dropout with prob 50 %
        for i in range(len(self.dpth_conv_l)):
            res = x
            for j in range(self.stack_size):
                x = functional.dropout(functional.relu(self.pt_conv_l[i][j](self.dpth_conv_l[i][j](x))), 0.4)
                x = self.batch_norm_l[i][j](x)
                if j == self.stack_size - 1:
                    x = functional.max_pool2d(x, 2)

                    x += self.res_conv_l[i](functional.max_pool2d(res, 2))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        return nn.LogSoftmax(dim=1)(self.fc1(x))
