"""The Vortex Detector Class and supporting modules.

This uses the Q-GAIN detector module to set up a detector suitable for detecting vortices.

"""
from __future__ import annotations

from copy import deepcopy

from qgain.detector import Detector

from scipy.ndimage import rotate
from skimage.filters import threshold_mean
from skimage.measure import label as meas_label
from skimage.measure import regionprops
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch


class VortexDetector(Detector):
    """Detect vortices in images of BECs.

    This uses the Q-GAIN library to set up a detector suitable for detecting vortices in images of BECs.

    Parameters
    ----------
    kwargs : dict
        Optional keywords to be passed to the object detection model initialization.

    """

    def __init__(self, **kwargs: dict) -> None:
        """Initialize a detector object suitable for vortices.

        Parameters
        ----------
        kwargs : dict
            Optional keywords to be passed to the object detection model initialization.

        """
        super().__init__(process_fn=vortex_process_fn,
                         od_model=ObjectDetector2D,
                         od_dataset_fn=VortexODDataset,
                         od_loss_fn=MetzLoss2D,
                         od_aug=True,
                         **kwargs)
        idx = self.controllers["ML Controller"].get_id(name="OD")
        self.controllers["ML Controller"].tools[idx]["tool"].metrics += [{"name": "Accuracy", "metric": accu_metric}]

    def use_models(self, model_paths: list, data: list | dict | None = None) -> None:
        """Use the vortex detector to make predictions.

        Parameters
        ----------
        model_paths : list
            The names of the saved weights. These should end in 'object.pt'.
        data : list or dict
            The target data to use the model on. By default this will be the data loaded into the detector object. It
            is possible to use external data by providing a list of dicts or dicts of dicts to this argument.
            (default = None)

        """
        super().use_models(model_list=("object detector",), model_paths=model_paths, data=data)
        target_data = self.data if data is None else data
        for item in target_data:
            if "OD_pred" in item:
                item["OD_pred"] = vortex_labels_to_data(item["OD_pred"][0], threshold=(0.5, 8))
        if data is None:
            self.data = target_data

    def plot_metrics(self, types: list | tuple = ("object detector"),
                     *, style: str | None = None, save: bool = False,
                     plot_kwargs: dict[dict] | None = None, data: list | dict | None = None) -> None:
        """Run plotting routines and display the results.

        Parameters
        ----------
        types : list
            The plotting tools to use.
            (default = ('object detector'))
        style : str
            An optional argument to specify a matplotlib style file and change the overall look of the plots.
            (default = None)
        save : bool
            An optional argument that will save the output rather than display it.
            (default = False)
        plot_kwargs : dict of dicts
            Optional arguments to be passed to the tool's callable function. This dictionary should contain the name of
            the plotting tool as a key with its value being the keyword dictionary to pass to the function.
            (default = None)
        data : list or dict
            The data to generate plots from. By default this is the data loaded into the detector object. If using a
            different target the function expects a similar structure to that of the SolDet module.
            (default = None)

        """
        plotter_kwargs = {"OD": {"ground_keys": ["positions"]}}
        if plot_kwargs is not None:
            plotter_kwargs.update(plot_kwargs)
        super().plot_metrics(types=types, style=style, save=save, plot_kwargs=plotter_kwargs, data=data)

    def vortex_counter(self, data: dict | list | np.ndarray, model_path: str) -> list:
        """Count the number of identified objects.

        Parameters
        ----------
        data : list or dict or numpy array
            The data to make predictions on.
        model_path : str
            The path to the saved weights for the model.

        Returns
        -------
        count_list : list
            A list of counts for each provided image.

        """
        if type(data) is np.ndarray:
            res = [{"data": deepcopy(data)}]
        elif type(data) is dict:
            res = [deepcopy(data)]
        else:
            res = deepcopy(data)

        self.use_models(model_paths=[model_path], data=res)

        count_list = []
        for item in res:
            count_list.append(len(item["OD_pred"]))

        return count_list


class ObjectCell(torch.nn.Module):
    """Object cell for use in 2D Object Detector.

    This cell represents the base layer of the 2D Object Detector used in identifying the probability of a vortex
    being present and its position.

    Parameters
    ----------
    in_channels: int
        The number of input channels in the image.
    out_channels: int
        The number of output channels in the image.
    kernel: tuple
        The 2D kernel size to use of shape (kH, kW)
    pool: list
        The size of the kernel to use during Max Pooling of the data.
        (default = None)
    dropout: float
        How often neurons should drop out.
        (default = 0.1)

    """

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple, pool: list | None = None,
                 dropout: float = 0.1) -> None:
        """Initialize a single cell of NN layers for the vortex OD.

        Parameters
        ----------
        in_channels: int
            The number of input channels in the image.
        out_channels: int
            The number of output channels in the image.
        kernel: tuple
            The 2D kernel size to use of shape (kH, kW)
        pool: list
            The size of the kernel to use during Max Pooling of the data.
            (default = None)
        dropout: float
            How often neurons should drop out.
            (default = 0.1)

        """
        super().__init__()

        # Network Layers
        self.conv_lay1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, padding="same", groups=in_channels, bias=False),
            torch.nn.Conv2d(in_channels, out_channels, 1, padding="same", groups=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels))
        self.conv_lay2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel, padding="same", groups=out_channels, bias=False),
            torch.nn.Conv2d(out_channels, out_channels, 1, padding="same", groups=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels))

        self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, padding="same", groups=1, bias=False)

        # Functional Layers
        self.dropout = torch.nn.Dropout(dropout)
        if pool is not None:
            self.pool = torch.nn.MaxPool2d(pool)
        else:
            self.pool = None

    def forward(self, x: Tensor) -> Tensor:
        """Perform a single pass through the network.

        Parameters
        ----------
        x: Tensor
            The target tensor to pass through the network

        Returns
        -------
        x: Tensor
            The result of the forward pass

        """
        x_dw = self.conv_lay1(x)
        x_dw = self.conv_lay2(x_dw)
        x = torch.add(x_dw, self.skip(x))
        x = self.dropout(x)

        return x if self.pool is None else self.pool(x)


class ObjectDetector2D(torch.nn.Module):
    """2D Vortex Object Detector.

    This pytorch object detector model identifies the position of excitations in two dimensions.
    Based on the work done in https://arxiv.org/abs/2012.13097.

    Parameters
    ----------
    layers : int
        The number of layers to use in the model. Each layer creates an object_cell with corresponding parameters.
        (default = 4)
    in_channels : list or tuple
        A list of input channels to the 2D convolutions for each layer.
        (default = (1, 8, 16, 32))
    out_channels : list or tuple
        A list of output channels to the 2D convolutions for each layer.
        (default = (8, 16, 32, 64))
    pool : list or tuple
        A list of pooling kernel sizes for each layer.
        (default = (None, (2,2), None, (2,2)))
    kernel : list or tuple
        The 2D kernel size to use of shape (kH, kW) for each layer.
        (default = (7, 7))
    label_shape : list or tuple
        The size of the position labels after converting from real positions to the compressed cell representation.
        (default = (33, 33))
    dropout : float
        How often neurons should drop out.
        (default = 0.1)

    """

    def __init__(self, dropout: float = 0.1, layers: int = 4, in_channels: list | tuple = (1, 8, 16, 32),
                 out_channels: list | tuple = (8, 16, 32, 64), pool: list | tuple = (None, (2, 2), None, (2, 2)),
                 kernel: list | tuple = (7, 7), label_shape: list | tuple = (33, 33)) -> None:
        """Initialize the model.

        Parameters
        ----------
        layers : int
            The number of layers to use in the model. Each layer creates an object_cell with corresponding parameters.
            (default = 4)
        in_channels : list or tuple
            A list of input channels to the 2D convolutions for each layer.
            (default = (1, 8, 16, 32))
        out_channels : list or tuple
            A list of output channels to the 2D convolutions for each layer.
            (default = (8, 16, 32, 64))
        pool : list or tuple
            A list of pooling kernel sizes for each layer.
            (default = (None, (2,2), None, (2,2)))
        kernel : list or tuple
            The 2D kernel size to use of shape (kH, kW) for each layer.
            (default = (7, 7))
        label_shape : list or tuple
            The size of the position labels after converting from real positions to the compressed cell representation.
            (default = (33, 33))
        dropout : float
            How often neurons should drop out.
            (default = 0.1)

        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for idx in range(layers):
            self.layers.append(ObjectCell(in_channels=in_channels[idx], out_channels=out_channels[idx],
                                          kernel=kernel, pool=pool[idx], dropout=dropout))

        # Output Layers
        self.pool = torch.nn.AdaptiveMaxPool2d(label_shape)
        self.final = torch.nn.Conv2d(in_channels=out_channels[-1], out_channels=out_channels[-1] * 2,
                                     kernel_size=kernel, padding="same")
        self.output = torch.nn.Conv2d(in_channels=out_channels[-1] * 2, out_channels=3, kernel_size=kernel,
                                      padding="same")
        self.output_act = torch.nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Take a tensor and identify any vortices and their positions.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        -------
        x : tensor of shape (B, 3, H // 4, W // 4)
            The output tensor containing the probabilities for a vortex to be present in one of the cells and its
            fractional position within a cell. Here B is the batch size and dimension 1 contains the probability (0),
            the vertical position in the cell (1), and the horizontal position within the cell (2). For the position
            values 0 to 1 indicate which side of the cell and by extension which pixel after conversion.
            Dimension 2 indicates the number of cells in the vertical direction and dimension 3 indicates the number of
            cells in the horizontal direction.

        """
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)

        return self.output_act(self.output(self.final(x)))


class MetzLoss2D(torch.nn.Module):
    """Loss function for vortex detection.

    Implementation of the loss function defined in: https://arxiv.org/abs/2012.13097
    The first term is essentially the weighted cross entropy probability for the cell belonging to the 'vortex present'
    class.
    The second term is a mean-squared error for the fractional position within the cell.

    Parameters
    ----------
    CE_weight : float
        The weight value used when calculating the cross entropy probability.
    MSE_weight : float
        The weight value used when calculating the mean-squared error.

    """

    def __init__(self, ce_weight: float = 3, mse_weight: float = 1) -> None:
        """Initialize the loss functionality."""
        super().__init__()
        self.CE_weight = ce_weight
        self.MSE_weight = mse_weight

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Do a single forward pass of the loss function.

        Parameters
        ----------
        prediction : tensor
            The input tensor representing the model output.
        target : tensor
            The input tensor representing the target label.

        Returns
        -------
        loss : tensor
            The result of the loss calculation.

        """
        eps = 1e-10
        ce_loss = (-self.CE_weight * target[:, 0, :, :] * torch.log(prediction[:, 0, :, :] + eps)
                   - (1 - target[:, 0, :, :]) * torch.log(1 - prediction[:, 0, :, :] + eps))
        mse_loss = (self.MSE_weight * target[:, 0, :, :] * ((target[:, 1, :, :] - prediction[:, 1, :, :])**2
                                                        + (target[:, 2, :, :] - prediction[:, 2, :, :])**2))

        return torch.sum(ce_loss + mse_loss) / prediction.shape[0]


class VortexODDataset(torch.utils.data.Dataset):
    """A dataset class for the ML based 2D object detector.

    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'. It will also grab the positions at key 'positions'.

    Parameters
    ----------
    data : list of dicts
            The data to build a dataset from.
    threshold : list or tuple
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
        value is considered the same excitation.
        (default = (0.5, 8))
    dims : list or tuple
        The shape of the image data.
        (default = (132, 132))
    augment: bool
        Specifies whether or not to augment the input data with rotations, translations, and flips.
        (default = False)

    """

    def __init__(self, data: list[dict], threshold: list | tuple = (0.5, 8), dims: list | tuple = (132, 132),
                 *, augment: bool = False) -> None:
        """Initialize the dataset functionality.

        Parameters
        ----------
        data : list of dicts
            The data to build a dataset from.
        threshold : list or tuple
            A list of values that influence the conversion between real positions and cell positions.
            Threshold[0] is the minimum value to consider an excitation is present.
            Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
            value is considered the same excitation.
            (default = (0.5, 8))
        dims : list or tuple
            The shape of the image data in the form (height, width).
            (default = (132, 132))
        augment: bool
            Specifies whether or not to augment the input data with rotations, translations, and flips.
            (default = False)

        """
        self.threshold = threshold
        self.dims = dims

        img_data = []
        label_data = []
        for entry in data:
            if entry["data"].shape[0] != entry["data"].shape[1]:
                msg = "Loaded image data is rectangular. Vortex Detector enforces square data."
                raise ValueError(msg)

            img_data.append(entry["data"])
            if "positions" in entry:
                label_data.append(entry["positions"])
            else:
                label_data.append([])

            if augment:
                # Rotation
                angle = np.random.default_rng().random() * 2 * np.pi
                rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
                aug_img = rotate(deepcopy(entry["data"]), angle * (180 / np.pi), reshape=False, order=2)
                img_data.append(aug_img)
                if "positions" in entry:
                    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    xy_rot = (deepcopy(np.array(entry["positions"])) - ((np.array(aug_img.shape) - 1) / 2)) @ rot
                    xy_rot += (np.array(aug_img.shape) - 1) / 2
                    label_data.append(xy_rot.tolist())
                else:
                    label_data.append([])

                # hflip of rotation
                img_data.append(np.flip(deepcopy(aug_img), 1))
                if "positions" in entry:
                    flip_pos = []
                    for coord in deepcopy(xy_rot.tolist()):
                        xy_flip = np.array(coord)
                        xy_flip[0] = (np.array(entry["data"].shape) - 1)[1] - xy_flip[0]
                        flip_pos.append(xy_flip.tolist())
                    label_data.append(flip_pos)
                else:
                    label_data.append([])

                # vflip of rotation
                img_data.append(np.flip(deepcopy(aug_img), 0))
                if "positions" in entry:
                    flip_pos = []
                    for coord in deepcopy(xy_rot.tolist()):
                        xy_flip = np.array(coord)
                        xy_flip[1] = (np.array(entry["data"].shape) - 1)[0] - xy_flip[1]
                        flip_pos.append(xy_flip.tolist())
                    label_data.append(flip_pos)
                else:
                    label_data.append([])

                # Translation
                max_tr_x = (entry["data"][entry["data"].shape[0] // 2,
                                             (entry["data"].shape[1] // 2):] == 0).nonzero()[0].shape[0]
                max_tr_x = np.min([max_tr_x, (entry["data"][entry["data"].shape[0] // 2,
                                                        :(entry["data"].shape[1] // 2)] == 0).nonzero()[0].shape[0]])
                max_tr_x = int(np.round(max_tr_x * 0.25))

                max_tr_y = (entry["data"][(entry["data"].shape[0] // 2):,
                                          (entry["data"].shape[1] // 2)] == 0).nonzero()[0].shape[0]
                max_tr_y = min(max_tr_y, (entry["data"][:(entry["data"].shape[0] // 2),
                                                        (entry["data"].shape[1] // 2)] == 0).nonzero()[0].shape[0])
                max_tr_y = int(np.round(max_tr_y * 0.25))

                for taugment, tpos in zip(deepcopy(img_data[-4:]), deepcopy(label_data[-4:]), strict=True):
                    trans_x = np.random.default_rng().integers(-max_tr_x, max_tr_x)
                    trans_y = np.random.default_rng().integers(-max_tr_y, max_tr_y)
                    img_data.append(np.roll(taugment, (trans_x, trans_y), (1, 0)))

                    if "positions" in entry:
                        trans_pos = []
                        for coord in tpos:
                            xy_trans = np.array(coord)
                            xy_trans[0] += trans_x
                            xy_trans[1] += trans_y
                            trans_pos.append(xy_trans.tolist())
                        label_data.append(trans_pos)
                    else:
                        label_data.append([])

                # Noise
                noise = np.random.default_rng().normal(0.5, 0.25, entry["data"].shape) * 0.05
                noise[entry["data"] == 0] = 0
                aug_img = deepcopy(entry["data"]) + noise
                img_data.append(aug_img)
                if "positions" in entry:
                    label_data.append(entry["positions"])
                else:
                    label_data.append([])

        self.imgs = torch.from_numpy(np.array(img_data)).float().unsqueeze(1)
        self.pos = torch.from_numpy(self.data_to_labels(label_data)).float()
        self.og_labels = label_data

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int

        """
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list]:
        """Retrieve a sample at the specified index.

        Parameters
        ----------
        idx : int
            The sample index

        Returns
        -------
        image : ndarray
            The image data at the specified index
        pos : list of floats
            A list of positions at the specified index

        """
        image = self.imgs[idx]
        pos = self.pos[idx]

        return image, pos

    def labels_to_data(self, label_out: np.ndarray) -> list[float]:
        """Convert the labels in cell space to positions in pixel space.

        Parameters
        ----------
        label_out : ndarray
            An array of probability and fractional position values in cell space.

        Returns
        -------
        labels : list
            positions in pixel space

        """
        if type(label_out) is not np.ndarray:
            msg = "Invalid type. Label data should be provided as numpy array."
            raise ValueError(msg)

        return vortex_labels_to_data(label_out, threshold=self.threshold)

    def data_to_labels(self, label_in: list) -> np.ndarray:
        """Convert the positions in pixel space to positions and probability in cell space.

        Parameters
        ----------
        label_in : list
            A list of position lists.

        Returns
        -------
        labels: ndarray
            The positions and probabilities in the cell space.

        """
        if type(label_in) is not list:
            msg = "Invalid type. Label data should be provided as a list."
            raise ValueError(msg)

        return vortex_data_to_labels(label_in, xdim=self.dims[1], ydim=self.dims[0])


def vortex_labels_to_data(label_in: np.ndarray, threshold: list) -> list:
    """Label conversion function.

    Convert between vortex positions in pixel space and cell space. This new space is a compressed representation of
    the positions in pixel space and the probability of them being present in a cell. The new space is a (3, 33, 33)
    array of values with the first (33, 33) entries representing the probability of an excitation being located in a
    cell, and the other two (33, 33) entries representing the fractional position of the excitation in that cell.
    Each cell represents a window of 4 x 4 pixels (H x W).

    This will convert from cell space to pixel space.

    Parameters
    ----------
    label_in : ndarray
        For each cell whose probability is above the threshold will have a position calculated. This position will be
        based on the fractional position in the cell. If multiple excitations exists next to each other and fall below
        the threshold the average positions will be calculated between the two.
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider that an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
        value is considered the same excitation.

    Returns
    -------
    label_out : list
        The list of positions in pixel space.

    """
    label_out = []
    if label_in.shape == (3, 33, 33):
        for i in range(33):
            for j in range(33):
                if label_in[0, j, i] > threshold[0]:
                    label_out.append([4 * i + 4 * label_in[1, j, i], 4 * j + 4 * label_in[2, j, i]])

        if len(label_out) > 1:
            i = 0
            while (i + 1) < len(label_out):
                j = i + 1
                while j < len(label_out):
                    dist = np.sqrt((label_out[j][0] - label_out[i][0])**2 + (label_out[j][1] - label_out[i][1])**2)
                    if dist < threshold[1]:
                        label_out[i][0] = (label_out[j][0] + label_out[i][0]) / 2
                        label_out[i][1] = (label_out[j][1] + label_out[i][1]) / 2
                        del label_out[j]
                    else:
                        j += 1
                i += 1

    elif label_in.shape[1:] == (3, 33, 33):
        for label in label_in:
            l_out = []
            for i in range(33):
                for j in range(33):
                    if label[0, j, i] > threshold[0]:
                        l_out.append([4 * i + 4 * label[1, j, i], 4 * j + 4 * label[2, j, i]])

            if len(l_out) > 1:
                i = 0
                while (i + 1) < len(l_out):
                    j = i + 1
                    while j < len(l_out):
                        dist = np.sqrt((l_out[j][0] - l_out[i][0])**2 + (l_out[j][1] - l_out[i][1])**2)
                        if dist < threshold[1]:
                            l_out[i][0] = (l_out[j][0] + l_out[i][0]) / 2
                            l_out[i][1] = (l_out[j][1] + l_out[i][1]) / 2
                            del l_out[j]
                        else:
                            j += 1
                    i += 1
            label_out.append(l_out)
    else:
        msg = f"Input has incorrect dimensions. Expected (N, 3, 33, 33) or (3, 33, 33) but got {label_in.shape}."
        raise ValueError(msg)

    return label_out


def vortex_data_to_labels(label_in: list, xdim: int = 132, ydim: int = 132) -> np.ndarray:
    """Label conversion function.

    Convert between vortex positions in pixel space and cell space. This new space is a compressed representation of
    the positions in pixel space and the probability of them being present in a cell. The new space is a (3, 33, 33)
    array of values with the first (33, 33) entries representing the probability of an excitation being located in a
    cell, and the other two (33, 33) entries representing the fractional position of the excitation in that cell.
    Each cell represents a window of 4 x 4 pixels (H x W).

    This will convert from pixel space to cell space.

    Parameters
    ----------
    label_in : list
        A list of positions in pixel space. Valid input can be a list of a single value for single image input, or a
        list of sub lists of positions for multiple images. The output will be an array of (3, 33, 33).
    xdim : int
        The horizontal dimension of the image data.
        (default = 132)
    ydim : int
        The vertical dimension of the image data.
        (default = 132)

    Returns
    -------
    label_out : ndarray
        Array of (3, 33, 33) values in cell space.

    """
    dims = [xdim, ydim]

    # Positions to Label
    label_out = np.zeros((3, 33, 33))
    if len(label_in) == 0:
        pass
    elif len(label_in) == 1:
        if len(label_in[0]) != 0:
            for coord in label_in[0]:
                if coord[0] < dims[0] and coord[0] > 0:
                    if coord[1] < dims[1] and coord[1] > 0:
                        # Probability a soliton is present
                        label_out[0, int(coord[1] // 4), int(coord[0] // 4)] = 1
                        # Fractional position along x direction of cell
                        label_out[1, int(coord[1] // 4), int(coord[0] // 4)] = (coord[0] % 4) / 4
                        # Fractional position along y direction of cell
                        label_out[2, int(coord[1] // 4), int(coord[0] // 4)] = (coord[1] % 4) / 4
                    else:
                        print("vortex positon beyond image dimensions.")
                else:
                    print("vortex positon beyond image dimensions.")

    elif len(label_in) > 1:
        label_out = np.zeros((len(label_in), 3, 33, 33))
        for i, pos in enumerate(label_in):
            if len(pos) != 0:
                for coord in pos:
                    if coord[0] < dims[0] and coord[0] > 0:
                        if coord[1] < dims[1] and coord[1] > 0:
                            # Probability a soliton is present
                            label_out[i, 0, int(coord[1] // 4), int(coord[0] // 4)] = 1
                            # Fractional position along x direction of cell
                            label_out[i, 1, int(coord[1] // 4), int(coord[0] // 4)] = (coord[0] % 4) / 4
                            # Fractional position along y direction of cell
                            label_out[i, 2, int(coord[1] // 4), int(coord[0] // 4)] = (coord[1] % 4) / 4
                        else:
                            print("vortex positon beyond image dimensions.")
                    else:
                        print("vortex positon beyond image dimensions.")

    return label_out


def vortex_process_fn(data_path: str, pos_path: str | None = None, tag: str | int = "unlabeled",
                      scale: list | None = None) -> list[dict]:
    """Process function for vortex data.

    This will load in a target numpy data file containing image arrays and prepare it for use by the vortex detector.
    Corresponding position labels can be provided which expects a numpy array containing position data for each entry in
    the image array data.

    Parameters
    ----------
    data_path : string
        The target file containing the image data. This should be a numpy file of arrays of image data of shape
        (N, H, W).
    pos_path : string
        The target file containing the position data for vortex locations. This should be a numpy object file of a
        list or array of N lists or tuples containing the X, Y positions for each vortex.
        (default = None)
    tag : string or int
        A descriptive label for the image.
        (default = 'unlabeled')
    scale : list or None
        Specifies the min max values to use for scaling the data to lay between 0 and 1.
        If None then no scaling is performed on the data.
        (default = None)

    Returns
    -------
    data_samples : list
        A list of dictionaries containing the collected pre-processed data

        Each dictionary contains, at minimum:
            The masked and unmasked image data of shape (132, 132).
            The tag value.
            The sub-directory.
            The original image size.
            The original data path.

        If positions were provided these are also saved in the dictionary entries.

    """
    data = np.load(data_path)
    labels = np.load(pos_path, allow_pickle=True).tolist() if pos_path is not None else None

    data_samples = []

    for (image, pos) in tqdm((zip(data, labels, strict=True)), desc="Processing vortex data..", total=data.shape[0]):
        sample = {}

        fullimgsize = image.shape
        yhigh = fullimgsize[0]
        xhigh = fullimgsize[1]

        if yhigh != xhigh:
            if xhigh > yhigh:
                crop = (xhigh - yhigh) // 2
                od_image = deepcopy(image[:, crop:-crop])
                yhigh = od_image.shape[0]
                xhigh = od_image.shape[1]
                positions = deepcopy(np.array(pos))
                positions[:, 0] -= crop
                positions = positions.tolist()
            else:
                crop = (yhigh - xhigh) // 2
                od_image = deepcopy(image[crop:-crop, :])
                yhigh = od_image.shape[0]
                xhigh = od_image.shape[1]
                positions = deepcopy(np.array(pos))
                positions[:, 1] -= crop
                positions = positions.tolist()
        else:
            od_image = deepcopy(image)
            positions = deepcopy(pos)

        if scale is not None:
            od_image = (od_image - scale[0]) / (scale[1] - scale[0])

        yc, xc = np.ogrid[:yhigh, :xhigh]
        regions = regionprops(meas_label(od_image > threshold_mean(od_image)))

        area = []
        for region in regions:
            area.append(region.area)
        area = np.asarray(area)
        region = regions[(area == np.max(area)).nonzero()[0][0]]

        ceny, cenx = region.centroid
        circ_r = region.axis_major_length / 2

        mask = np.sqrt((xc - cenx)**2 + (yc - ceny)**2) <= circ_r
        cropped_od_image = od_image * mask

        sample["Original Data Size"] = fullimgsize
        sample["cloud_data"] = od_image
        sample["data"] = cropped_od_image
        sample["tag"] = tag
        sample["sub_dir"] = "Vortex_" + str(tag)
        if pos is not None:
            # re-sort to make the position ordering match that of the labeling function
            labels = vortex_data_to_labels([positions], xdim=fullimgsize[1], ydim=fullimgsize[0])
            sample["positions"] = vortex_labels_to_data(labels, threshold=[0.5, 8])

        data_samples += [sample]

    return data_samples


def accu_metric(pred: Tensor, targ: Tensor) -> float:
    """Accuracy metric for vortex object detector.

    This checks the probability cell for the presence of an excitation and tracks how many are correct.

    Parameters
    ----------
    pred : Tensor
        The output tensor of the model
    targ : Tensor
        The target labels

    Returns
    -------
    correct : float
        The accuracy of correctly identified vortices.

    """
    batch_correct = 0
    threshold = 0.5
    for idx, prediction in enumerate(pred):
        if (targ[idx, 0] > threshold).any():
            num_exc = (targ[idx, 0] == torch.max(targ[idx, 0])).nonzero().shape[0]
            t_loc = targ[idx, 0].flatten() > threshold
            p_loc = prediction[0].flatten() > threshold
            batch_correct += (torch.logical_and(t_loc, p_loc).nonzero().shape[0] / num_exc)
        elif (prediction[0] < threshold).all():
            batch_correct += 1

    return batch_correct / pred.shape[0]
