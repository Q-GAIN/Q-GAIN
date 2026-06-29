"""Object model for SolDet."""
from qgain.models import ObjectDetector

from torch import Tensor
from torch import log as tlog
from torch import sum as tsum
from torch.nn import Module


class MetzLoss(Module):
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

    def forward(self, pred: Tensor, tars: Tensor) -> Tensor:
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
        loss = tsum(-self.param * tars[:, 0, :, :] * tlog(pred[:, 0, :, :] + eps)
                         - (1 - tars[:, 0, :, :]) * tlog(1 - pred[:, 0, :, :] + eps)
                         + self.param * tars[:, 0, :, :] * (tars[:, 1, :, :] - pred[:, 1, :, :])**2)

        return loss / pred.shape[0]


class ObjectDetector(ObjectDetector):
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
        super().__init__(out_channels=128, filter_list=[8, 16, 32, 64], stack_size=2, kernel_size=7,
                         cell_dim=label_shape, dimension="W", dropout=0.5)

    def forward(self, x: Tensor) -> Tensor:
        """Take a tensor and identify any solitons and their positions.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, 2, 1, W // 4)
            The output tensor containing the probabilities for a soliton to be present in one of the cells and its
            fractional position within a cell. Here B is the batch size and dimension 1 contains the probability (0)
            and the position within the cell (1), where values 0 to 1 indicate left to right. Dimension 2 indicates the
            number of cells in the vertical direction and dimension 3 indicates the number of cells in the horizontal
            direction.

        """
        return super().forward(x)
