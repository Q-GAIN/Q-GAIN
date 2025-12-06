"""Dataset class definitions for preparing data to be used by their respective models."""
from __future__ import annotations

import contextlib
import zipfile
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from torchvision.transforms.functional import hflip, vflip
from tqdm import tqdm

from qgain.utilities import config


class SolitonPIEClassDataset(torch.utils.data.Dataset):
    """A dataset class for the physics informed classifier.

    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'.
    It will also grab the positions at key 'positions'.

    Parameters
    ----------
    data : list or dict
        The data to build a dataset from.

    """

    def __init__(self, data: list | dict) -> None:
        """Initialize the dataset class.

        Parameters
        ----------
        data : list or dict
            The data to build a dataset from.

        """
        x = []
        y = []
        for sample in data:
            if sample["label"] == 1:
                if sample["data"].shape[0] == sample["data"].shape[1]:
                    msg = "Loaded image data is square. 1D SolDet module enforces rectangular data."
                    raise ValueError(msg)
                x.append(sample["data"])
                y.append(sample["positions"])

        self.img_data = x
        self.pos = y

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int

        """
        return len(self.img_data)

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
        return self.img_data[idx], self.pos[idx]


class SolitonQEClassDataset(torch.utils.data.Dataset):
    """A dataset class for the physics informed quality estimate.

    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'.
    It will also grab the positions at key 'positions'.

    Parameters
    ----------
    data : list or dict
        The data to build a dataset from.

    """

    def __init__(self, data: list | dict) -> None:
        """Initialize the dataset class.

        Parameters
        ----------
        data : list or dict
            The data to build a dataset from.

        """
        x = []
        y = []
        for sample in data:
            if "excitation_PIE" in sample:
                if sample["data"].shape[0] == sample["data"].shape[1]:
                    msg = "Loaded image data is square. 1D SolDet module enforces rectangular data."
                    raise ValueError(msg)
                if sample["label"] == 1 and sample["excitation_PIE"] == [0]:
                    x.append(sample["data"])
                    y.append(sample["positions"])

        self.img_data = x
        self.pos = y

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int

        """
        return len(self.img_data)

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
        return self.img_data[idx], self.pos[idx]


class SolitonClassDataset(torch.utils.data.Dataset):
    """A dataset class for the ML based classifier.

    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'.
    It will also grab the class label at key 'label'.

    Parameters
    ----------
    data : list or dict
        The data to build a dataset from.
    augment : bool
        If set to True the data is augmented with rotations.
        (default = True)

    """

    def __init__(self, data: list, *, augment: bool = True) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        data : list or dict
            The data to build a dataset from.
        augment : bool
            If set to True the data is augmented with rotations.
            (default = True)

        """
        x = []
        y = []
        for entry in data:
            x.append(entry["data"])
            if entry["data"].shape[0] == entry["data"].shape[1]:
                msg = "Loaded image data is square. 1D SolDet module enforces rectangular data."
                raise ValueError(msg)

            if "label" not in entry:
                msg = "Data must be labeled with class numbers."
                raise ValueError(msg)
            y.append(entry["label"])

        x = np.array(x)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        y = np.array(y)

        if augment:
            x, y = augment_expand_as_mlst2021(x, y)

        self.imgs = torch.from_numpy(x).float()
        self.img_labels = torch.from_numpy(y)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int

        """
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Retrieve a sample at the specified index.

        Parameters
        ----------
        idx : int
            The sample index

        Returns
        -------
        image : ndarray
            The image data at the specified index
        label : int
            The class label at the specified index

        """
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label.long()

    @staticmethod
    def accu_metric(pred: torch.Tensor, targ: torch.Tensor) -> float:
        """Calculate the number of correct predictions and give an accuracy score.

        This is used during training of the SolDet CL to determine its accuracy during validation.

        Paramters
        ---------
        pred : torch.Tensor
            The output tensor of the object detector.
        targ : torch.Tensor
            The target label in cell space.

        Returns
        -------
        correct : float

        """
        return (pred.argmax(1) == targ).to(torch.float).mean().item()


class SolitonODDataset(torch.utils.data.Dataset):
    """A dataset class for the ML based object detector.

    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'.
    It will also grab the positions at key 'positions'.

    Parameters
    ----------
    data : list or dict
        The data to build a dataset from.
    augment : bool
        If set to True the data is augmented with rotations.
        (default = True)
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.

            - Threshold[0] is the minimum value to consider a soliton is present.
            - Threshold[1] is the minimum distance two solitons can be considered separate. Any distances under this
                value is considered the same excitation.

        (default = [0.5, 4])

    """

    def __init__(self, data: list | dict, threshold: list | tuple = (0.5, 4), *, augment: bool = True) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        data : list or dict
            The data to build a dataset from.
        augment : bool
            If set to True the data is augmented with rotations.
            (default = True)
        threshold : list
            A list of values that influence the conversion between real positions and cell positions.

                - Threshold[0] is the minimum value to consider a soliton is present.
                - Threshold[1] is the minimum distance two solitons can be considered separate. Any distances under this
                    value is considered the same excitation.

            (default = [0.5, 4])

        """
        self.threshold = threshold

        x = []
        y = []
        for entry in data:
            if entry["data"].shape[0] == entry["data"].shape[1]:
                msg = "Loaded image data is square. 1D SolDet enforces rectangular data."
                raise ValueError(msg)

            x.append(entry["data"])
            if "positions" in entry:
                y.append(entry["positions"])
            else:
                y.append([])

        x = np.array(x)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))

        if augment:
            x, y = augment_w_pos(x, y)
        else:
            x = torch.from_numpy(x).float()

        y = pos_41labels_conversion(y, threshold)

        self.imgs = x
        self.pos = torch.from_numpy(y).float()

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

    def labels_to_data(self, sample: np.ndarray) -> list:
        """Convert the labels in cell space to positions in pixel space.

        Parameters
        ----------
        sample : ndarray
            An array of probability and fractional position values in cell space.

        Returns
        -------
        labels : list
            The positions in pixel space

        """
        return pos_41labels_conversion(sample, threshold=self.threshold)

    @staticmethod
    def accu_metric(pred: torch.Tensor, targ: torch.Tensor) -> float:
        """Calculate the number of correct predictions and give an accuracy score.

        This is used during training of the SolDet OD to determine its accuracy during validation.

        Paramters
        ---------
        pred : torch.Tensor
            The output tensor of the object detector.
        targ : torch.Tensor
            The target label in cell space.

        Returns
        -------
        correct : float

        """
        batch_correct = 0
        for idx, prediction in enumerate(pred):
            if (targ[idx, 0] > 0.5).any():
                num_exc = (targ[idx, 0] == torch.max(targ[idx, 0])).nonzero().shape[0]
                t_loc = targ[idx, 0].flatten() > 0.5
                p_loc = prediction[0].flatten() > 0.5
                batch_correct += (torch.logical_and(t_loc, p_loc).nonzero().shape[0] / num_exc)
            elif (prediction[0] < 0.5).all():
                batch_correct += 1

        return batch_correct / pred.shape[0]


def augment_w_pos(images: np.ndarray, positions: list) -> tuple[torch.Tensor, list]:
    """Augment the data by rotating the images three ways: horizontal flip, vertical flip, and a 180 degree rotation.

    Parameters
    ----------
    images : ndarray
        The array of image data to be augmented. Typically (N, H, W) where N is the number of images, W is the width
        of the image and H is the height.
    positions : list
        A list of positions for each image in the images array.

    Returns
    -------
    aug_x : ndarray
        The augmented image data
    aug_y : list
        The augmented position data

    """
    x = torch.from_numpy(images).float()
    aug_x = x
    aug_y = positions.copy()
    xdim = x.shape[3]

    # hflip
    aug_x = torch.cat((aug_x, hflip(x)), 0)
    for pos in positions:
        if len(pos) == 0:
            aug_y.append([])
        elif len(pos) == 1:
            tmp = [xdim - pos[0]]
            aug_y.append(tmp)

    # vflip
    aug_x = torch.cat((aug_x, vflip(x)), 0)
    for pos in positions:
        aug_y.append(pos)

    # 180
    aug_x = torch.cat((aug_x, vflip(hflip(x))), 0)
    for pos in positions:
        if len(pos) == 0:
            aug_y.append([])
        elif len(pos) == 1:
            tmp = [xdim - pos[0]]
            aug_y.append(tmp)

    return aug_x, aug_y


def expand_data_by_augment(images: np.ndarray, labels: np.ndarray, augments: list) -> tuple[np.ndarray, np.ndarray]:
    """Support function for augment_expand_as_mlst2021.

    This does a simple horizontal, vertical, and 180 degree rotation.

    Parameters
    ----------
    images : ndarray
        The array of image data to be augmented. Typically (N, H, W) where N is the number of images, W is the width
        of the image and H is the height.
    labels : ndarray
        An array of class lebels for each image in the images array.
    augments : list
        A list of augments to apply to the data. Choices are 'hflip' for horizontal flipping, 'vflip' for vertical
        flipping, and '180rot' for a 180 degree rotation.

    Returns
    -------
    aug_x : ndarray
        The augmented image data
    aug_y : ndarray
        The augmented class label data

    """
    aug_x = [images]
    aug_y = [labels]
    if "hflip" in augments:
        aug_x.append(images[:, :, ::-1])
        aug_y.append(labels)

    if "vflip" in augments:
        aug_x.append(images[:, ::-1])
        aug_y.append(labels)

    if "180rot" in augments:
        aug_x.append(np.rot90(images, k=2, axes=(1, 2)))
        aug_y.append(labels)

    return np.concatenate(aug_x), np.concatenate(aug_y)


def augment_expand_as_mlst2021(raw_x: np.ndarray, raw_y: np.ndarray,
                               seed: int | np.typing.ArrayLike | np.random.SeedSequence | np.random.BitGenerator |
                               np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """Augment data suitable for classifiers in the SolDet module.

    This does a simple horizontal, vertical, and 180 degree rotation.

    Parameters
    ----------
    raw_x : ndarray
        The array of image data to be augmented. Typically (N, H, W) where N is the number of images, W is the width
        of the image and H is the height.
    raw_y : ndarray
        An array of class labels for each image in the images array.
    seed : int or array_like[ints] or SeedSequence or BitGenerator or Generator
        The seed to use to initialize the randomization generator
        (default = None)

    Returns
    -------
    augment_x : ndarray
        The augmented image data
    augment_y : ndarray
        The augmented class label data

    """
    rng = np.random.default_rng(seed=seed)
    zero_aug_x, zero_aug_y = expand_data_by_augment(raw_x[(raw_y == 0) | (raw_y == 2)],
                                                    raw_y[(raw_y == 0) | (raw_y == 2)],
                                                    augments=["hflip", "vflip", "180rot"])

    # select 1/3 each of one class to apply each transformation to.
    # if the raw_x/y were shuffled then these will also be shuffled
    idx = rng.permutation(list(range(raw_y[raw_y == 1].shape[0])))
    one_raw_x = raw_x[raw_y == 1][idx]
    one_raw_y = raw_y[raw_y == 1][idx]
    one_hflip_x, one_hflip_y = expand_data_by_augment(
        one_raw_x[::3], one_raw_y[::3], augments=["hflip"])
    one_vflip_x, one_vflip_y = expand_data_by_augment(
        one_raw_x[1::3], one_raw_y[1::3], augments=["vflip"])
    one_180_x, one_180_y = expand_data_by_augment(
        one_raw_x[2::3], one_raw_y[2::3], augments=["180rot"])

    augment_x = np.concatenate([zero_aug_x, one_hflip_x, one_vflip_x, one_180_x])
    augment_y = np.concatenate([zero_aug_y, one_hflip_y, one_vflip_y, one_180_y])
    return augment_x, augment_y


def pos_to_41labels_conversion(label_in: list) -> np.ndarray:
    """Convert soliton positions in pixel space to cell space.

    This new space is a compressed representation of the positions in pixel space and the probability of them being
    present in a cell. The new space is a (2, 1, 41) array of values with the first 41 entries representing the
    probability of an excitation being located in a cell, and the second 41 entries representing the fractional position
    of the excitation in that cell. Each cell represents 4 pixels in length, so each cell essentially represents a
    window of 132 x 4 pixels (H x W).

    Parameters
    ----------
    label_in : list
        A list of positions in pixel space. Valid input can be a list of a single value for single image input, or a
        list of sub lists of positions for multiple images. The output will be an array of (2, 1, 41).

    Returns
    -------
    label_out : ndarray
        An array of (2, 1, 41) values in cell space.

    """
    if label_in == []:
        label_out = np.zeros((2, 1, 41))
    elif type(label_in[0]) in {float, np.float64}:  # Postions on Single image
        label_out = np.zeros((2, 1, 41))
        for sub_l_in in label_in:
            if sub_l_in < 164 and sub_l_in > 0:
                label_out[0, 0, int(sub_l_in // 4)] = 1
                label_out[1, 0, int(sub_l_in // 4)] = (sub_l_in % 4) / 4
            else:
                print("soliton position beyond [0, 164].")

    elif type(label_in[0]) is list:  # A list of postions on many images
        label_out = np.zeros((len(label_in), 2, 1, 41))
        for i, pos in enumerate(label_in):
            for sub_l_in in pos:
                if sub_l_in < 164 and sub_l_in > 0:
                    label_out[i, 0, 0, int(sub_l_in // 4)] = 1
                    label_out[i, 1, 0, int(sub_l_in // 4)] = (sub_l_in % 4) / 4
                else:
                    print("soliton position beyond [0, 164].")
    return label_out


def labels_to_pos_conversion(label_in: np.ndarray, threshold: list | tuple = (0.5, 4)) -> list:
    """Convert soliton positions in cell space to pixel space.

    This new space is a compressed representation of
    the positions in pixel space and the probability of them being present in a cell. The new space is a (2, 1, 41)
    array of values with the first 41 entries representing the probability of an excitation being located in a cell,
    and the second 41 entries representing the fractional position of the excitation in that cell.
    Each cell represents 4 pixels in length, so each cell essentially represents a window of 132 x 4 pixels (H x W).

    The behavior of this function depends on the data type of label_in.

    Parameters
    ----------
    label_in : list or ndarray
        An array of values in cell space. For each cell whose probability is above the threshold it will have a position
        calculated. This position will be based on the fractional position in the cell. If multiple excitations exists
        next to each other the average positions will be calculated between the two.
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider that an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered separate. Any distances under this
        value is considered the same excitation.
        (default = [0.5, 4)])

    Returns
    -------
    label_out : list
        A list of positions in pixel space.

    """
    label_out = []
    if label_in.shape == (2, 1, 41):  # Single 41 label
        for i in range(41):
            if label_in[0, 0, i] > threshold[0]:
                label_out.append(4 * i + 4 * label_in[1, 0, i])
        if len(label_out) > 1:
            i = 0
            while (i + 1) < len(label_out):
                if (label_out[i + 1] - label_out[i]) < threshold[1]:
                    label_out[i] = (label_out[i + 1] + label_out[i]) / 2
                    del label_out[i + 1]
                else:
                    i += 1

    elif label_in.shape[1:] == (2, 1, 41):  # Array of 41 labels
        for label in label_in:
            l_out = []
            for i in range(41):
                if label[0, i, 0] > threshold[0]:
                    l_out.append(4 * i + 4 * label[1, 0, i])
            if len(l_out) > 1:
                i = 0
                while (i + 1) < len(l_out):
                    if (l_out[i + 1] - l_out[i]) < threshold[1]:
                        l_out[i] = (l_out[i + 1] + l_out[i]) / 2
                        del l_out[i + 1]
                    else:
                        i += 1
            label_out.append(l_out)
    else:
        msg = "Invalid input shape."
        raise ValueError(msg)
    return label_out


def pos_41labels_conversion(label_in: list | np.ndarray, threshold: list | tuple = (0.5, 4)) -> np.ndarray | list:
    """Convert between soliton positions in pixel space and cell space.

    This new space is a compressed representation of the positions in pixel space and the probability of them being
    present in a cell. The new space is a (2, 1, 41) array of values with the first 41 entries representing the
    probability of an excitation being located in a cell, and the second 41 entries representing the fractional position
    of the excitation in that cell. Each cell represents 4 pixels in length, so each cell essentially represents a
    window of 132 x 4 pixels (H x W).

    The behavior of this function depends on the data type of label_in.

    Parameters
    ----------
    label_in : list or ndarray
        If the data type is a list then it is assumed that this is a list of positions in pixel space. Valid input can
        be a list of a single value for single image input, or a list of sub lists of positions for multiple images.
        The output will be an array of (2, 1, 41).

        If the data type is an array then it is assumed this input is an array of values in cell space. For each cell
        whose probability is above the threshold it will have a position calculated. This position will be based on the
        fractional position in the cell. If multiple excitations exists next to each other the average positions will be
        calculated between the two.
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider that an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered separate. Any distances under this
        value is considered the same excitation.
        (default = [0.5, 4)])

    Returns
    -------
    label_out : list or ndarray
        If label_in was a list then the output is an array of (2, 1, 41) values in cell space.
        If label_in was an array then the output is a list of positions in pixel space.

    """
    if type(label_in) is list:  # if input is soliton positions
        label_out = pos_to_41labels_conversion(label_in=label_in)

    elif type(label_in) is np.ndarray:  # if input is 41 labels
        label_out = labels_to_pos_conversion(label_in=label_in, threshold=threshold)

    return label_out


def download_ds() -> None:
    """Download the public SolDet data set.

    This will expand the compressed files into the currently set experimental folder.
    """
    data_path, _ = config()

    roster_npy_exists = data_path.joinpath("data", "data_info", "data_roster.npy").is_file()
    roster_h5_exists = data_path.joinpath("data", "data_info", "data_roster.h5").is_file()
    if not roster_npy_exists and not roster_h5_exists:
        print("Downloading SolDet data. This may take a while. Please wait..")

        urls = ["https://data.nist.gov/od/ds/mds2-2363/data_info.zip",
                "https://data.nist.gov/od/ds/ark:/88434/mds2-2363/data_files.zip"]
        files = ["data_info.zip", "data_files.zip"]
        subdir = data_path.joinpath("data")
        for url, file in zip(urls, files):
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                response.raise_for_status()
                msg = f"{url} returned status code {response.status_code}."
                raise RuntimeError(msg)
            file_size = int(response.headers.get("Content-Length", 0))
            with subdir.joinpath(file).open("wb") as f, tqdm.wrapattr(response.raw, "read", total=file_size,
                                                                      desc=f"Downloading {file}.") as raw:
                chunk = raw.read(1024)
                if chunk:
                    f.write(chunk)
                while chunk:
                    chunk = raw.read(1024)
                    if chunk:
                        f.write(chunk)
            print("Extracting data. Please wait..")
            with zipfile.ZipFile(subdir.joinpath(file), "r") as z:
                z.extractall(subdir)
            subdir.joinpath(file).unlink()
    else:
        print("Data already exists. Skipping...")


def soldet_to_h5(path: str, *, delete_old: bool = True) -> None:
    """Convert the original SolDet dataset into the new h5 version.

    Parameters
    ----------
    path: str
        The path to the destination folder. This should contain the data and data_info folders of the SolDet module.
    delete_old : bool
        If true this will delete the old files when creating the new ones.
        (default = True)

    """
    target = Path(path)
    data_roster_dir = target.joinpath("data", "data_info")
    if not data_roster_dir.is_dir():
        msg = f"{data_roster_dir} is an invalid data_roster path."
        raise FileNotFoundError(msg)

    data_roster = {}
    try:
        roster_orig = np.load(data_roster_dir.joinpath("data_roster.npy"), allow_pickle=True).item()
    except FileNotFoundError as err:
        if data_roster_dir.joinpath("data_roster.h5").is_file():
            msg = "Roster file data_roster.npy not found, but data_roster.h5 was found. Did you already convert?"
            raise UserWarning(msg) from err
        msg = "Roster file data_roster.npy  not found."
        raise FileNotFoundError(msg) from err
    data_roster = {**data_roster, **roster_orig}
    if delete_old:
        data_roster_dir.joinpath("data_roster.npy").unlink()

    roster_path = data_roster_dir.joinpath("data_roster.h5")
    if roster_path.is_file():
        mode = "a"
        with h5py.File(roster_path, mode) as h5_file:
            i = len(h5_file.keys())
    else:
        mode = "w"
        i = 0

    for sample in tqdm(data_roster, desc="Converting data.."):
        label = None
        sample_name = target.name + f"_{i}"
        data_dir = target.joinpath("data", "data_files")
        numpy_file = Path(sample).name
        with contextlib.suppress(KeyError):
            label = data_roster[sample]["label_v3"] if label is None else label
        with contextlib.suppress(KeyError):
            label = data_roster[sample]["label_v2"] if label is None else label
        with contextlib.suppress(KeyError):
            label = data_roster[sample]["label_v1"] if label is None else label
        if label is None:
            msg = "Invalid label data."
            raise ValueError(msg)
        class_dir = Path(f"class-{label}")
        soldet_h5 = data_dir.joinpath(class_dir, sample_name + ".h5")

        with h5py.File(roster_path, mode, meta_block_size=8000) as h5_file:
            ds = h5_file.create_dataset(sample_name, data=h5py.Empty("f"), dtype="f", shape=None)
            ds.attrs["label"] = label
            ds.attrs["original_file"] = sample
            ds.attrs["path"] = str(class_dir.joinpath(sample_name + ".h5"))

        data_loaded = np.load(data_dir.joinpath(class_dir, numpy_file), allow_pickle=True).item()
        if delete_old:
            data_dir.joinpath(class_dir, numpy_file).unlink()

        with h5py.File(soldet_h5, "w") as h5_file:
            h5_file.attrs["label"] = label
            h5_file.attrs["original_file"] = Path(sample).name
            for item in data_roster[sample]:
                if "file_name" in item:
                    pass
                else:
                    h5_file.attrs.create(item, data_roster[sample][item])
            ds = h5_file.create_dataset("cloud_data", data=data_loaded["cloud_data"], compression="gzip",
                                        compression_opts=6)
            ds = h5_file.create_dataset("data", data=data_loaded["masked_data"], compression="gzip",
                                        compression_opts=6)

            for key in data_loaded:
                if str(key) == "masked_data" or str(key) == "cloud_data":
                    pass
                elif isinstance(data_loaded[key], dict):
                    ds = h5_file.create_dataset(str(key), data=h5py.Empty("f"), dtype="f", shape=None)
                    for subkey in data_loaded[key]:
                        ds.attrs[str(subkey)] = data_loaded[key][subkey]
                elif isinstance(data_loaded[key], np.ndarray):
                    ds = h5_file.create_dataset(str(key), data=data_loaded[key], compression="gzip",
                                                compression_opts=6)
                else:
                    h5_file.attrs[str(key)] = data_loaded[key]
        i += 1
        if mode == "w":
            mode = "a"
