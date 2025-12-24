"""Utility functions for the functioning of Q-GAIN."""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage as snd
from lmfit import Model, Parameters

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def combine_data_probe_bg(raw_data: np.ndarray) -> np.ndarray:
    """Generate naive OD image.

    A part of preprocessing. Combine 3 raw images (atom, probe, dark) into single atom image.

    Parameters
    ----------
    raw_data : ndarray
        Raw data with shape (3, H, W) [(atoms, probe, background)].

    Returns
    -------
    naive_OD : ndarray
        Optical depth of the image, the pixel values represent atom density.

    """
    probedark = raw_data[1] - raw_data[2]
    probedark[probedark == 0] = 1e-8
    absorbed_fraction = (raw_data[0] - raw_data[2]) / probedark
    count_max = int(np.nanmax(raw_data[1]))
    dark_var = np.nanvar(raw_data[2])
    odmax_meaningful = -np.log(np.sqrt(dark_var + count_max) / count_max)

    absorbed_fraction[absorbed_fraction <= 0] = 1e-3
    naive_od = -np.log(absorbed_fraction)
    naive_od[naive_od < -1.0] = -1.0
    naive_od[naive_od > (odmax_meaningful + 1)] = odmax_meaningful + 1

    return naive_od


def get_cloud_fit(naive_od: np.ndarray, angle: float, *, adjust_angle: bool = False) -> dict:
    """Thomas 2D Fermi fit.

    A part of preprocessing. Fits a 2D ThomasFermi envelope to the given atom cloud image.

    Parameters
    ----------
    naive_od : ndarray
        Full atom density image.
    angle : int or float.
        The angle between camera and atom cloud elongated direction.
    adjust_angle : boolean.
        Determine whether angle should be fitted.
        (default = False)

    Returns
    -------
    cloud_fit : dict
        A dictionary containing the best results of the fit. This dictionary will contain the following parameters:

            - 'amp': Amplitude (peak density) of the 2D cloud.
            - 'cenx': x coordinate of the center position of the cloud.
            - 'ceny': y coordinate of the center position of the cloud.
            - 'rx': The cloud width in x direction.
            - 'ry': The cloud width in y direction.
            - 'offset': The offset value of the fitting.
            - 'theta': Same as the input angle, but in radian.

    """
    fullimgsize = naive_od.shape
    ylow = 0
    yhigh = fullimgsize[0]
    xlow = 0
    xhigh = fullimgsize[1]

    xroi = np.arange(xlow, xhigh)
    yroi = np.arange(ylow, yhigh)
    x, y = np.meshgrid(xroi, yroi)

    x1d_distribution = np.sum(naive_od, 0)
    y1d_distribution = np.sum(naive_od, 1)

    _, peaksposx = _pickpeak(x1d_distribution, 5)
    _, peaksposy = _pickpeak(y1d_distribution, 5)

    thomas_fermi_2d_rotmodel = Model(thomas_fermi_2d_rot)

    pars = Parameters()
    pars.add("amp", value=2.0, vary=True)
    pars.add("cenx", value=np.mean(peaksposx) + xlow, vary=True)
    pars.add("ceny", value=np.mean(peaksposy) + ylow, vary=True)
    pars.add("rx", value=66, vary=True)
    pars.add("ry", value=56, vary=True)
    pars.add("offset", value=np.min(naive_od), vary=True)
    pars.add("theta", value=np.radians(angle), vary=adjust_angle)

    fit_tf_2d = thomas_fermi_2d_rotmodel.fit(naive_od.ravel(), params=pars, xy=(x, y))

    return fit_tf_2d.best_values


def _pickpeak(x: ArrayLike, npicks: int = 20) -> tuple[ArrayLike | Any, int]:
    """Pick npicks number of peaks from the distribution, largest to smallest.

    Support function used during fitting.

    Parameters
    ----------
    x : ArrayLike
        1D distribution of data.
    npicks : int
        The number of peaks to select from the data.
        (default = 20)

    Returns
    -------
    vals : ArrayLike
        The values of the selected peaks.
    idx : ArrayLike
        The indices of the peaks.

    """
    # sort array and take index
    idx = np.argsort(-x)  # inverse of sort array--take the maximum value first

    idx = idx[0:npicks]
    vals = x[idx]

    return vals, idx


def thomas_fermi_2d_rot(xy: tuple, amp: float, cenx: float, ceny: float, rx: float, ry: float, offset: float,
                     theta: float) -> np.ndarray:
    """Fit to a ThomasFermi 2D model.

    ThomasFermi 2D fitting for use in preprocessing atom cloud images.
    This function supports get_cloud_fit and does the actual fitting of the 2D Thomas Fermi envelope.

    Parameters
    ----------
    xy: tuple
        The positions from a meshgrid of points.
    amp: float
        Amplitude (peak density) of the 2D cloud.
    cenx: float
        x coordinate of the center position of the cloud.
    ceny: float
        y coordinate of the center position of the cloud.
    rx: float
        The cloud width in x direction.
    ry: float
        The cloud width in y direction.
    offset: float
        The offset value of the fitting.
    theta: float
        Same as the input angle, but in radian.

    Returns
    -------
    F(x, y) : ndarray
        The result of the function.

    """
    x, y = xy
    xx = (x - cenx) * np.cos(theta) + (y - ceny) * np.sin(theta)
    yy = (y - ceny) * np.cos(theta) - (x - cenx) * np.sin(theta)

    b = 1 - (xx / rx)**2 - (yy / ry)**2
    b = np.maximum(b, 0)
    tf2d = amp * (b**(3 / 2)) + offset

    return tf2d.ravel()


def rotate_crop(naive_od: np.ndarray, cloud_fit: dict, xdim: int, ydim: int) -> np.ndarray:
    """Rotate and crop the image.

    A part of preprocessing. Given an image and fit parameters, rotate and crop the image to emphasize the atom cloud.

    Parameters
    ----------
    naive_od : ndarray
        Image to be processed. Pixel values represent atom density.
    cloud_fit : dict
        TF2D fitting parameters of the cloud. See get_cloud_fit function.
    xdim : int
        Width of the image.
    ydim : int
        Height of the image.

    Returns
    -------
    roi : ndarray
        The cropped and rotated atom cloud image.

    """
    x_size = xdim // 2
    y_size = ydim // 2

    center = np.array([cloud_fit["cenx"], cloud_fit["ceny"]])
    angle_rad = cloud_fit["theta"]
    atoms_rot, pt_rot = rotate_img(naive_od, center, angle_rad)

    if pt_rot[0] + x_size > (atoms_rot.shape[1] - 1):
        right_length = (atoms_rot.shape[1] - 1) - pt_rot[0]
        left_length = xdim - right_length

    elif pt_rot[0] - x_size < 0:
        left_length = pt_rot[0]
        right_length = xdim - left_length

    else:
        left_length = x_size
        right_length = x_size

    if pt_rot[1] + y_size > (atoms_rot.shape[0] - 1):
        top_length = (atoms_rot.shape[0] - 1) - pt_rot[1]
        bottom_length = ydim - top_length

    elif pt_rot[1] - y_size < 0:
        bottom_length = pt_rot[1]
        top_length = ydim - bottom_length

    else:
        top_length = y_size
        bottom_length = y_size

    xroi_crop = np.arange(pt_rot[0] - left_length, pt_rot[0] + right_length)
    yroi_crop = np.arange(pt_rot[1] - bottom_length, pt_rot[1] + top_length)

    roi = atoms_rot[yroi_crop, :]

    return roi[:, xroi_crop]


def rotate_img(image: np.ndarray, point: int, angle_rad: float) -> tuple[np.ndarray, int]:
    """Rotate an image (clockwise) and a selected point within the image by a given angle.

    Parameters
    ----------
    image : ndarray
        Image to be processed.
    point : int
        The point to rotate.
    angle_rad : float
        The angle to rotate through.

    Returns
    -------
    im_rot : ndarray
        The rotated image.
    new_point : int
        The new position of the point.

    """
    im_rot = snd.rotate(image, np.degrees(angle_rad), reshape=True)
    org_center = (np.array(image.shape[:2][::-1]) - 1) / 2
    rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2

    tr = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                  [-np.sin(angle_rad), np.cos(angle_rad)]])
    new_point = tr.reshape(2, 2) @ (point - org_center) + rot_center

    return im_rot, new_point.astype(int)


def apply_mask(crop_rotated_data: np.ndarray, cloud_fit: dict, orig_size: tuple) -> np.ndarray:
    """Apply elliptical mask to data to remove background.

    Parameters
    ----------
    crop_rotated_data : ndarray
        Atom cloud density image.
    cloud_fit : dict
        TF2D fitting parameters of the cloud. See get_cloud_fit function.
    orig_size: tuple
        The original shape of the image data.

    Returns
    -------
    masked_data : ndarray
        Atom cloud density image with mask.

    """
    # extract image size
    imgsize = crop_rotated_data.shape
    # extract rotation angle
    angle_rad = cloud_fit["theta"]

    # extract three points defining the ellipse
    x0, y0 = cloud_fit["cenx"], cloud_fit["ceny"]
    xx, yx = point_pos(x0, y0, d=cloud_fit["rx"], angle_rad=angle_rad)
    xy, yy = point_pos(x0, y0, d=cloud_fit["ry"], angle_rad=angle_rad + np.deg2rad(90))

    points = [(x0, y0), (xx, yx), (xy, yy)]

    # rotate the ellipse
    rot_array = snd.rotate(np.zeros(orig_size), np.deg2rad(angle_rad), reshape=True)
    rot_pts = rotate_mask(points, angle_rad, orig_size, rot_array.shape)

    # overlay the ellipse on the rotated empty array (array of zeros)
    mask_rot = in_ellipse(rot_array, rot_pts)

    if int(rot_pts[0][1]) - int(imgsize[0] / 2) < 0:
        bot_idx = 0
        top_idx = int(rot_pts[0][1]) + int(imgsize[0] / 2) + np.abs(int(rot_pts[0][1]) - int(imgsize[0] / 2))

    elif int(rot_pts[0][1]) + int(imgsize[0] / 2) > mask_rot.shape[0]:
        bot_idx = int(rot_pts[0][1]) - int(imgsize[0] / 2)
        bot_idx -= ((int(rot_pts[0][1]) + int(imgsize[0] / 2)) - mask_rot.shape[0])
        top_idx = mask_rot.shape[0]

    else:
        bot_idx = int(rot_pts[0][1]) - int(imgsize[0] / 2)
        top_idx = int(rot_pts[0][1]) + int(imgsize[0] / 2)

    if int(rot_pts[0][0]) - int(imgsize[1] / 2) < 0:
        left_idx = 0
        right_idx = int(rot_pts[0][0]) + int(imgsize[1] / 2) + np.abs(int(rot_pts[0][0]) - int(imgsize[1] / 2))

    elif int(rot_pts[0][0]) + int(imgsize[1] / 2) > mask_rot.shape[1]:
        left_idx = int(rot_pts[0][0]) - int(imgsize[1] / 2)
        left_idx -= ((int(rot_pts[0][0]) + int(imgsize[1] / 2)) - mask_rot.shape[1])
        right_idx = mask_rot.shape[1]

    else:
        left_idx = int(rot_pts[0][0]) - int(imgsize[1] / 2)
        right_idx = int(rot_pts[0][0]) + int(imgsize[1] / 2)

    mask_final = mask_rot[bot_idx:top_idx, left_idx:right_idx]

    return crop_rotated_data * mask_final


def rotate_mask(points: list, angle_rad: float, mask_shape: tuple, mask_rot_shape: tuple) -> list[int]:
    """Rotate (clockwise) a list of points within an image by a given angle.

    Parameters
    ----------
    points: list
        A list of points to rotate.
    angle_rad: float
        The angle to rotate through.
    mask_shape: tuple
        The original shape of the applied mask.
    mask_rot_shape: tuple
        The shape of the rotated data.

    Returns
    -------
    points_rot : list of int
        Rotated image with mask.

    """
    org_center = (np.array(mask_shape[:2][::-1]) - 1) / 2
    rot_center = (np.array(mask_rot_shape[:2][::-1]) - 1) / 2

    tr = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                  [-np.sin(angle_rad), np.cos(angle_rad)]])

    return [(tr.reshape(2, 2) @ (point - org_center) + rot_center).astype(int) for point in points]


def point_pos(x0: float, y0: float, d: float, angle_rad: float) -> tuple[int, int]:
    """Find coordinates of a point d distance away from (x0, y0) at an angle.

    Parameters
    ----------
    x0: float
        Reference point in x.
    y0: float
        Reference point in x.
    d: float
        The distance from the points.
    angle_rad: float
        The angle to find the distance from.

    Returns
    -------
    x0 : int
        The new coordinate in x.
    y0 : int
        The new coordinate in y.

    """
    return int(x0 + d * np.cos(angle_rad)), int(y0 + d * np.sin(angle_rad))


def in_ellipse(arr: np.ndarray, pts: list) -> np.ndarray:
    """Check which point within an array lay inside a defined ellipse.

    Parameters
    ----------
    arr: ndarray
        The array of data to be checked.
    pts: list
        The parameters that define the ellipse in the form [ellipsis_center, vertex, co-vertex].

    Returns
    -------
    arr : ndarray
        A boolean array indicating which points are within the ellipse.

    """
    rx = pts[1][0] - pts[0][0]
    ry = pts[2][1] - pts[0][1]

    return np.array([(x - pts[0][0])**2 / rx**2 + (y - pts[0][1])**2 / ry**2 <= 1
                     for y in range(arr.shape[0])
                        for x in range(arr.shape[1])]).reshape(arr.shape)


def config() -> tuple[Path, str]:
    """Configure Q-GAIN.

    Sets up the configuration file to be used by the Q-GAIN library with default values if it doesn't exist.
    Also creates the required data folders specified by the configuration file if the folders are not found.
    By default this is in the user's HOME path when a folder path is not specified.

    Returns
    -------
    exp_path : Path
        A pathlib object pointing to where the experimental folders for Q-GAIN are located.
    exp_name : str
        The name of the currently set experiment.

    """
    qgain_path = Path(__file__).parent
    user_path = qgain_path.joinpath("CONFIG.ini")

    if not user_path.is_file():
        print("Warning: No configuration file found. Creating.")
        home_dir = Path.home()
        configfile = configparser.ConfigParser()
        configfile["PATHS"] = {"data_path": str(home_dir.joinpath("qgain")), "def_exp_name": "default_ds"}
        configfile.write(user_path.open("w"))

    configfile = configparser.ConfigParser()
    configfile.read(user_path)
    data_path = configfile["PATHS"]["data_path"]
    exp_name = configfile["PATHS"]["def_exp_name"]
    exp_path = Path(data_path)
    exp_path = exp_path.joinpath(exp_name)

    if not exp_path.is_dir():
        print("User path not found, creating.")
        exp_path.mkdir(parents=True)

    if not exp_path.joinpath("data").is_dir():
        exp_path.joinpath("data").mkdir(parents=True)

    if not exp_path.joinpath("models").is_dir():
        exp_path.joinpath("models").mkdir(parents=True)

    top_dir = ["data_files", "data_info"]

    for directory in top_dir:
        if not exp_path.joinpath("data", directory).is_dir():
            exp_path.joinpath("data", directory).mkdir(parents=True)

    return exp_path, exp_name


def change_exp(value: str) -> None:
    """Change the current experiment folder in Q-GAIN.

    Parameters
    ----------
    value: str
        The name of the experiment to switch to.

    """
    qgain_path = Path(__file__).parent
    user_path = qgain_path.joinpath("CONFIG.ini")
    if not user_path.is_file():
        msg = "No configuration file found."
        raise FileNotFoundError(msg)

    configfile = configparser.ConfigParser()
    configfile.read(user_path)
    configfile["PATHS"]["def_exp_name"] = value
    with user_path.open("w") as file:
        configfile.write(file)


def change_path(value: str) -> None:
    """Change the path to where the experimental folders for Q-GAIN are kept.

    Parameters
    ----------
    value: str
        The path of the destination folder.

    """
    qgain_path = Path(__file__).parent
    user_path = qgain_path.joinpath("CONFIG.ini")
    if not user_path.is_file():
        msg = "No configuration file found."
        raise FileNotFoundError(msg)

    configfile = configparser.ConfigParser()
    configfile.read(user_path)
    configfile["PATHS"]["data_path"] = value
    with user_path.open("w") as file:
        configfile.write(file)
