"""IO function definitions.

This file supports the input/output of Q-GAIN.

"""
from __future__ import annotations

from pathlib import Path
import random

from qgain.utilities import apply_mask, combine_data_probe_bg, get_cloud_fit, rotate_crop

from tqdm import tqdm
import h5py
import numpy as np


def get_raw_data(directory: str, target: str, atoms_name: str, bg_name: str, probe_name: str,
                 meta_list: list | tuple = (), *, return_files_names: bool = False, return_metadata: bool = False,
                  shuffle: bool = False) -> list:
    """Given a directory of Labscript experimental h5 files, obtain the image data based on the supplied naming schemes.

    Parameters
    ----------
    directory : string
        The target directory of h5 files.
    target : string
        The directory name in the h5 file containing the cloud images.
    atoms_name : string
        The full or partial name for images containing the atoms, probe, and background.
    bg_name : string
        The full or partial name for images of only the background, no atoms or probe light.
    probe_name : string
        The full or partial name for images of only the probe light.
    return_files_names : boolean
        If True, appends the list of filenames to the returned list of data.
        (default = False)
    return_metadata : boolean
        If True, appends the metadata list of the files to the returned list of data.
        (default = False)
    meta_list : list or tuple
        An optional list of metadata attributes to retrieve from the globals folder of a labscript h5 file.
        These are appended to the data list as a separate list if return_metadata is True.
        (default = ())
    shuffle : boolean
        If True, shuffles the order of h5 files found in the supplied directory.
        (default = False)

    Returns
    -------
    res : list
        A list of tuples containing the image data for the three TOF files.
        By default the resulting shape of the return is a list of N entries of 3xWxH for (atoms, probe, background).

        If return_metadata is True an additional sub list of N entries is added containing the specified labscript
        globals.
        The resulting shape of the return is a list of sublists of N entries, with one containing the meta data.

        If return_files_names is True an additional sub list of N entries is added containing the filenames of found
        entries.
        The resulting shape of the return is a list of sublists of N entries, with one containing the filenames.

    """
    if type(shuffle) is int:
        if_shuffle = True
        seed = shuffle
    elif shuffle:
        if_shuffle = True
        seed = None
    else:
        if_shuffle = False

    directory = Path(directory)
    files = list(directory.rglob("*.h5"))
    if if_shuffle:
        random.Random(seed).shuffle(files)

    datasize = len(files)

    raw_data_list = []
    if return_metadata:
        metadata_list = []

    for file in tqdm(files[:datasize], desc="Getting Raw Data.."):
        with h5py.File(file, "r") as h5_file:
            g = h5_file["images/" + target]
            p_path = g.visit(lambda x: x if probe_name in x else None)
            b_path = g.visit(lambda x: x if bg_name in x else None)
            a_path = g.visit(lambda x: x if atoms_name in x else None)
            if (p_path is None) or (b_path is None) or (a_path is None):
                pass
            else:
                probe = g[p_path]
                atoms = g[a_path]
                background = g[b_path]
                img = (atoms, probe, background)
                img = np.float64(img)
                raw_data_list.append(img)
                if return_metadata:
                    sub_list = []
                    for meta in meta_list:
                        if meta in h5_file["globals"].attrs:
                            sub_list.append((meta, h5_file["globals"].attrs[meta]))
                    metadata_list.append(sub_list)

    res = [raw_data_list]
    if return_metadata:
        res.append(metadata_list)
    if return_files_names:
        res.append([file.name for file in files])

    if len(res) == 1:
        return res[0]

    return res


def process_data(path: str, target: str, atoms_name: str, bg_name: str, probe_name: str,
                 width: int, height: int, tag: int | str, *, camera_angle: float = 0, meta_list: list | tuple = (),
                 labels: dict | None = None, return_metadata: bool = False,
                 return_files_names: bool = True) -> list[dict]:
    """Obtain image data, meta data, and filenames and then pre-process it for use in Q-GAIN.

    Given a directory of labscript experimental h5 files, this obtains image data, meta data, and filenames and then
    pre-processes it for use in Q-GAIN.
    From the images a basic OD is calculated and used for a 2D ThomasFermi fit.
    The cloud is then cropped and masked.
    For each OD image all data is saved as a dictionary.

    Parameters
    ----------
    path : string
        The target directory of h5 files.
    target : string
        The directory name in the h5 file containing the images.
    atoms_name : string
        The full or partial name for images containing the atoms.
    bg_name : string
        The full or partial name for images of only the background, no atoms or probe light.
    probe_name : string
        The full or partial name for images of only the probe light.
    width : int
        The target width of the images after processing.
    height : int
        The target height of the images after processing.
    tag : int or string
        A descriptive label for the data.
    camera_angle : float
        The angle between the camera and the elongated axis of the atom cloud, if any.
        (default = 0)
    return_files_names: boolean
        If True, appends the list of filenames to the returned list of data.
        (default = True)
    return_metadata: boolean
        If True, appends the metadata specified in the meta_list argument to the returned list of data.
        (default = False)
    meta_list : list
        An optional list of metadata attributes to retrieve from the globals folder of a labscript h5 file.
        These are appended to the data list as a separate list if return_metadata is True.
        (default = ())
    labels: dict or None
        An optional list or tuple of dictionaries holding additional descriptive labels for the data. The keys and their
        values are copied to the sample's dictionary entry.
        (default = None)

    Returns
    -------
    data_samples : list
        A list of dictionaries containing the collected pre-processed data.
        Each dictionary contains, at minimum:

            - The data of shape (height, width).
            - A descriptive tag of the data.
            - The sub directory it should be stored in.
            - The 2D TF fit parameters.
            - The rotation angle.
            - The original image size.
            - The original file name.

        Any optional meta data is also saved if return_metadata is True.

    """
    if not Path(path).is_dir():
        msg = "Invalid path provided."
        raise FileNotFoundError(msg)

    raw_data = get_raw_data(path, target, atoms_name, bg_name, probe_name, return_metadata=return_metadata,
                            meta_list=meta_list, return_files_names=return_files_names, shuffle=False)

    data_list = raw_data[0] if return_metadata or return_files_names else raw_data

    data_samples = []
    for i in tqdm(range(len(data_list)), desc="Processing Raw Data.."):
        sample = {}
        img = data_list[i]
        if return_metadata and not return_files_names:
            for item in raw_data[1][i]:
                sample[item[0]] = item[1]
        elif return_files_names and not return_metadata:
            sample["filename"] = raw_data[1][i]
        elif return_metadata and return_files_names:
            for item in raw_data[1][i]:
                sample[item[0]] = item[1]
            sample["filename"] = raw_data[2][i]

        sample["Original Data Size"] = img[0].shape
        naive_od = combine_data_probe_bg(img)
        full_image_fit = get_cloud_fit(naive_od, camera_angle, adjust_angle=False)
        sample["rot_angle"] = full_image_fit["theta"]
        sample["fitted_parameters"] = full_image_fit
        cloud_data = rotate_crop(naive_od, full_image_fit, xdim=width, ydim=height)
        sample["cloud_data"] = cloud_data
        sample["data"] = apply_mask(cloud_data, full_image_fit, img[0].shape)
        sample["tag"] = tag
        sample["sub_dir"] = f"class-{tag}"
        if labels is not None:
            sample.update(labels)
        data_samples += [sample]

    return data_samples


def load_data(path: str, tags: list, *, minmax: list | None = None, scale: bool = False) -> list[dict]:
    """Load data from the sub directories listed in the roster file of the currently set experimental folder.

    Parameters
    ----------
    path : string
        The path to the experimental folder.
    tags : list
        The data to load. Labels specified here will load all files in the corresponding folder.
    scale : boolean
        If True the data will be scaled so it is bounded between 0 and 1.
        (default = False)
    minmax : list
        If scale is set to True the data will be scaled given the minimum and maximum values specified in minmax.
        This expects [MIN, MAX], if none is previded then the global minimum and maximum values found in the set are
        used.
        (default = [None, None])

    Returns
    -------
    data_roster : list
        A list of dictionaries containing the loaded data.

    """
    if minmax is not None:
        min_val = minmax[0]
        max_val = minmax[1]
    else:
        min_val = np.inf
        max_val = -np.inf

    roster_path = Path(path).joinpath("data/data_info")
    data_path = Path(path).joinpath("data/data_files")
    if not roster_path.is_dir():
        msg = f"{roster_path} is an invalid data_roster path."
        raise FileNotFoundError(msg)
    roster_path = roster_path.joinpath("data_roster.h5")

    data_roster = []
    targets = []
    with h5py.File(roster_path, "r") as h5_file:
        for sample in h5_file:
            if h5_file[sample].attrs["tag"] in tags:
                targets.append(sample)

        for sample in tqdm(targets, desc="Loading processed data.."):
            with h5py.File(data_path.joinpath(h5_file[sample].attrs["path"]), "r") as sample_file:
                data_sample = {}
                attr_keys = list(sample_file.attrs.keys())
                sample_keys = list(sample_file.keys())
                try:
                    if "_dtype" in sample_file["data"].attrs and sample_file["data"].attrs["_dtype"] == "dict":
                        # This is a dictionary.
                        data_sample["data"] = {}
                        for item in sample_file["data"].attrs:
                            data_sample["data"][item] = sample_file["data"].attrs[item]
                    elif "_dtype" in sample_file["data"].attrs and sample_file["data"].attrs["_dtype"] == "list":
                        # This is a list.
                        if h5py.check_string_dtype(sample_file["data"].dtype) is not None:
                            data_sample["data"] = sample_file["data"].asstr()[()].tolist()
                        else:
                            data_sample["data"] = sample_file["data"][()].tolist()
                    elif h5py.check_string_dtype(sample_file["data"].dtype) is not None:
                        # This is a string.
                        data_sample["data"] = sample_file["data"].asstr()[()]
                    else:
                        data_sample["data"] = sample_file["data"][()]
                    sample_keys.remove("data")

                    if minmax is None and scale:
                        min_val = np.min([min_val, np.min(data_sample["data"])])
                        max_val = np.max([max_val, np.max(data_sample["data"])])

                    data_sample["tag"] = sample_file.attrs["tag"]
                    attr_keys.remove("tag")

                    for attr in attr_keys:
                        data_sample[str(attr)] = sample_file.attrs[attr]

                    for key in sample_keys:
                        if sample_file[key][()].shape is None:
                            # Dictionary
                            data_sample[str(key)] = {}
                            for item in sample_file[key].attrs:
                                data_sample[str(key)][item] = sample_file[key].attrs[item]
                        elif h5py.check_string_dtype(sample_file[key].dtype) is not None:
                            # string
                            data_sample[str(key)][item] = sample_file[str(key)].asstr()[()]
                        else:
                            # Array
                            data_sample[str(key)] = sample_file[key][()]
                    data_sample["path"] = h5_file[sample].attrs["path"]

                    data_roster.append(data_sample)
                except Exception as e:
                    if type(e) is KeyboardInterrupt or type(e) is MemoryError or type(e) is SystemExit:
                        raise
                    tqdm.write("Error. Skipping entry because: " + str(e))

    min_val = np.floor(min_val) if minmax is None and scale else min_val
    max_val = np.ceil(max_val) if minmax is None and scale else max_val

    if scale:
        for sample in tqdm(data_roster, desc="Normalizing Data.."):
            sample["data"] = (sample["data"] - min_val) / (max_val - min_val)

    return data_roster
