"""Base neural network architectures that a user can sub-class for their own purposes."""

from torch import Tensor, nn
from torch import flatten as tflatten
from torch.cuda import is_available
from torch.special import expit

device = "cuda" if is_available() else "cpu"


class MLST2021CNNmodern(nn.Module):
    """Base classification model.

    A base classification model that can be used for classification tasks. The model is split into layers of stacks.
    Each stack contains a depthwise convolution, a pointwise convolution, and a batch normalization operation. Each
    layer contains a residual connection to bypass the stacks via a residual convolution in the last stack of a layer.
    During the forward pass at the end of each layer a max pooling operation is applied which divides the H and W
    dimensions of the input by 2.

    After all layers are completed an adaptive average pooling is applied before the output is flattened. This is then
    fed to a linear layer to transform the data into the number of specified classes. Final output is the logarithm of
    the softmax function.

    Parameters
    ----------
    filter_list: list of ints
        A list of channel sizes for the convolutional layers. These values will be the output channel sizes for the
        depthwise convolution operations. These will also be the input and output channels for the pointwise
        convolution operations. The number of items here dictates the number of layers in the model.
    stack_size: int
        The number of convolution stacks within each model layer. Each single stack applies a depthwise
        convolution, pointwise convolution, and a batch normalization. The last stack will contain a residual
        convolution connection to provide a residual pathway between model layers.
    kernel_size: int or tuple
        The kernel size to use for the depthwise convolution.
    num_classes : int
        The number of classes to identify.

    """

    def __init__(self, filter_list: list[int], stack_size: int, kernel_size: int | tuple, num_classes: int,
                 dropout: float = 0.5) -> None:
        """Initialize model.

        Parameters
        ----------
        filter_list : list of ints
            A list of channel sizes for the convolutional layers. These values will be the output channel sizes for the
            depthwise convolution operations. These will also be the input and output channels for the pointwise
            convolution operations. The number of items here dictates the number of layers in the model.
        stack_size : int
            The number of convolution stacks within each model layer. Each single stack applies a depthwise
            convolution, pointwise convolution, and a batch normalization. The last stack will contain a residual
            convolution connection to provide a residual pathway between model layers.
        kernel_size : int or tuple
            The kernel size to use for the depthwise convolution.
        num_classes : int
            The number of classes to identify.
        dropout : float
            The amount of dropout to use during training.
            (default = 0.5)

        """
        super().__init__()

        self.stack_size = stack_size
        self.dropout = dropout
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
                self.dpth_conv_l[i].append(nn.Conv2d(block_in_channels, f, kernel_size, padding="same",
                                                     groups=block_in_channels))
                self.pt_conv_l[i].append(nn.Conv2d(f, f, 1))
                self.batch_norm_l[i].append(nn.BatchNorm2d(f))

                if j == self.stack_size - 1:
                    self.res_conv_l.append(nn.Conv2d(in_channels, f, 1))

                block_in_channels = f
            in_channels = f

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(filter_list[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Take a tensor and make a class prediction.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, num_classes)
            The output tensor containing the probabilities for each class. The output is of shape (B, num_classes) where
            B is the batch size and the last dimension contains the probabilities for the number of classes specified in
            the model initialization.

        """
        for i in range(len(self.dpth_conv_l)):
            res = x
            for j in range(self.stack_size):
                x = nn.functional.dropout(nn.functional.relu(self.pt_conv_l[i][j](self.dpth_conv_l[i][j](x))),
                                          self.dropout)
                x = self.batch_norm_l[i][j](x)
                if j == self.stack_size - 1:
                    x = nn.functional.max_pool2d(x, 2)

                    x += self.res_conv_l[i](nn.functional.max_pool2d(res, 2))

        x = self.adaptive_pool(x)
        x = tflatten(x, 1)

        return nn.LogSoftmax(dim=1)(self.fc1(x))


class ObjectDetector(nn.Module):
    """Base 1D object detection model.

    A base object detection model that can be used for detection tasks. The model is split into layers of stacks.
    Each stack contains a depthwise convolution, a pointwise convolution, and a batch normalization operation. Each
    layer contains a residual connection to bypass the stacks via a residual convolution in the last stack of a layer.

    The model will detect along the H or W axis based on the setting of the "dimension" parameter. At the end of each
    layer a max pooling is applied. Each pooling operation results in a shape of (H // 4, W // 2) when dimension = W and
    (H // 2, W // 4) when dimension = H. The second to last layer applies a different pooling kernel which results in a
    final output of (H // 4, W // 1) when dimension = W and (H // 1, W // 4) when dimension = H. The final layer will
    apply an adaptive average pooling to make the (H, W) dimensions match that of the cell space.

    The model outputs in a cell space that is specified by a probability of an object being present and its fractional
    position in the cell. The number of cells in this space is specified by the "label_shape" parameter. It is expected
    that the cell space dimensions match the axis detection is done along.

    The final output will be the logistic sigmoid function of the final convolutions. These convolutions take the output
    of the previous model layers and transform the tensors into shape (B, 2, h, w) where h and w are the cell space
    dimensions and B is the batch size.

    Parameters
    ----------
    out_channels : int
        The final output channel size after applying all model layers. A convolution will be applied whose input
        channels are the last value in the filter_list parameter and its output channels are out_channels. A kernel of
        (1, kernel_size) or (kernel_size, 1) will be applied depending on if the dimension parameter is "W" or "H",
        respectively.
    filter_list : list of ints
        A list of channel sizes for the convolutional layers. These values will be the output channel sizes for the
        depthwise convolution operations. These will also be the input and output channels for the pointwise
        convolution operations. The number of items here dictates the number of layers in the model.
    stack_size : int
        The number of convolution stacks within each model layer. Each single stack applies a depthwise
        convolution, pointwise convolution, and a batch normalization. The last stack will contain a residual
        convolution connection to provide a residual pathway between model layers.
    kernel_size : int or tuple
        The kernel size to use for the depthwise convolution.
    dimension : str
        The dimension to do object detection along. This expects a value of "W" for detecting along the horizontal or
        "H" for detecting along the vertical.
    cell_dim : tuple of ints
        The shape of the cell space. This should match the setting specified in the dimension parameter so that the
        value is (1, C) for "W" and (C, 1) for "H", where C is the number of cells.
    dropout : float
        The amount of dropout to use during training.
        (default = 0.5)

    """

    def __init__(self, out_channels: int, filter_list: list[int], stack_size: int, kernel_size: int | tuple,
                 cell_dim: tuple[int, int] | list[int, int], dimension: str, dropout: float = 0.5) -> None:
        """Initialize the model.

        Parameters
        ----------
        out_channels : int
            The final output channel size after applying all model layers. A convolution will be applied whose input
            channels are the last value in the filter_list parameter and its output channels are out_channels. A kernel
            of (1, kernel_size) or (kernel_size, 1) will be applied depending on if the dimension parameter is "W" or
            "H", respectively.
        filter_list : list of ints
            A list of channel sizes for the convolutional layers. These values will be the output channel sizes for the
            depthwise convolution operations. These will also be the input and output channels for the pointwise
            convolution operations. The number of items here dictates the number of layers in the model.
        stack_size : int
            The number of convolution stacks within each model layer. Each single stack applies a depthwise
            convolution, pointwise convolution, and a batch normalization. The last stack will contain a residual
            convolution connection to provide a residual pathway between model layers.
        kernel_size : int or tuple
            The kernel size to use for the depthwise convolution.
        dimension : str
            The dimension to do object detection along. This expects a value of "W" for detecting along the horizontal
            or "H" for detecting along the vertical.
        cell_dim : tuple of ints
            The shape of the cell space. This should match the setting specified in the dimension parameter so that the
            value is (1, C) for "W" and (C, 1) for "H", where C is the number of cells.
        dropout : float
            The amount of dropout to use during training.
            (default = 0.5)

        """
        super().__init__()

        self.stack_size = stack_size
        self.dropout = dropout
        self.dimension = dimension
        in_channels = 1
        block_in_channels = in_channels

        self.dpth_conv_l = nn.ModuleList()
        self.pt_conv_l = nn.ModuleList()
        self.batch_norm_l = nn.ModuleList()
        self.res_conv_l = nn.ModuleList()
        for i, f in enumerate(filter_list):
            self.dpth_conv_l.append(nn.ModuleList())
            self.pt_conv_l.append(nn.ModuleList())
            self.batch_norm_l.append(nn.ModuleList())

            for j in range(self.stack_size):
                if i == len(filter_list) - 1 and self.dimension == "W":
                    self.dpth_conv_l[i].append(nn.Conv2d(block_in_channels, f, (1, kernel_size), padding="same",
                                                         groups=block_in_channels))
                elif i == len(filter_list) - 1 and self.dimension == "H":
                    self.dpth_conv_l[i].append(nn.Conv2d(block_in_channels, f, (kernel_size, 1), padding="same",
                                                         groups=block_in_channels))
                else:
                    self.dpth_conv_l[i].append(nn.Conv2d(block_in_channels, f, (kernel_size, kernel_size),
                                                         padding="same", groups=block_in_channels))
                self.pt_conv_l[i].append(nn.Conv2d(f, f, 1))
                self.batch_norm_l[i].append(nn.BatchNorm2d(f))

                if j == self.stack_size - 1:
                    self.res_conv_l.append(nn.Conv2d(in_channels, f, 1))

                block_in_channels = f
            in_channels = f

        self.adaptive_pool = nn.AdaptiveAvgPool2d(cell_dim)
        if self.dimension == "W":
            self.pconv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding="same")
            self.fconv = nn.Conv2d(out_channels, 2, (1, kernel_size), padding="same")
        elif self.dimension == "H":
            self.pconv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding="same")
            self.fconv = nn.Conv2d(out_channels, 2, (kernel_size, 1), padding="same")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Take a tensor and identify any solitons and their positions.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, 2, cell_dim)
            The output tensor containing the probabilities for a soliton to be present in one of the 41 cells and its
            fractional position within a cell. Here B is the batch size and dimension 1 contains the probability (0)
            and the position within the cell (1), where values 0 to 1 indicate left to right. Dimensions 2 and 3 will
            reflect the number of cells in the vertical and horizontal directions.

        """
        for i in range(len(self.dpth_conv_l)):
            res = x
            for j in range(self.stack_size):
                x = nn.functional.relu(nn.functional.dropout(self.pt_conv_l[i][j](self.dpth_conv_l[i][j](x)),
                                                             self.dropout))
                x = self.batch_norm_l[i][j](x)
                if j == self.stack_size - 1:
                    if self.dimension == "W":
                        kernel = (4, 1) if i == len(self.dpth_conv_l) - 2 else (4, 2)
                    else:
                        kernel = (1, 4) if i == len(self.dpth_conv_l) - 2 else (2, 4)

                    if i == len(self.dpth_conv_l) - 1:
                        x = self.adaptive_pool(x)
                        x += self.res_conv_l[i](self.adaptive_pool(res))
                    else:
                        x = nn.functional.max_pool2d(x, kernel)
                        x += self.res_conv_l[i](nn.functional.max_pool2d(res, kernel, kernel))

        x = nn.functional.relu(nn.functional.dropout(self.pconv(x), self.dropout))

        return expit(self.fconv(x))
