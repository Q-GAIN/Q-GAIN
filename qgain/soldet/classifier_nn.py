"""Classifier model for SolDet."""
from qgain.models import MLST2021CNNmodern

from torch import Tensor


class SolDetClassifier(MLST2021CNNmodern):
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
        super().__init__(filter_list=[8, 16, 32, 64, 128], stack_size=3, kernel_size=8, num_classes=num_classes,
                         dropout=0.4)

    def forward(self, x: Tensor) -> Tensor:
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
            batch size and the last dimension contains the probabilities for the number of classes
            specified in the model initialization.

        """
        return super().forward(x)
