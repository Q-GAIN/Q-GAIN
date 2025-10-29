"""PI functionality for PIE and QE metrics."""
from __future__ import annotations

import numpy as np
from lmfit import Model, Parameters
from scipy.optimize import curve_fit, fmin
from tqdm import tqdm

from qgain.utilities import _pickpeak


def fit_soliton(vec_x: np.ndarray, roixwithoutbackg: np.ndarray, res: dict, inti_pos: list | None = None,
                func: str = "gaussian1D", n: int = 1, *, return_fit_curve: bool = False,
                return_list: bool = False) -> (tuple[np.ndarray, list] | list) | (dict | tuple[np.ndarray, dict]):
    """Fit solitons to a 1D profile with the given function.

    Parameters
    ----------
    vec_x : 1D numpy array
        Pixel index list.
    roixwithoutbackg : 1D numpy array
        1D profile. The pixel values with the Thomas Fermi 1D Sum fit subtracted.
    res: dictionary
        The fit parameters from the Thomas Fermi 1D Sum fit.
    inti_pos : list
        The initial positions of the solitonic excitations. Used to set the 'cen' and 'amp' parameters.
        (default = None)
    func : string
        The fitting function used for the fit. Choices are gaussian, original (SOLDET 1.0), or modern (SOLDET 2.0).
        (default = 'gaussian1D')
    n : integer
        The number of excitations in the region of interest.
        (default = 1)
    return_fit_curve : boolean
        If True, return the fitting curve. Return will be (curve fit, fit values).
        If False, the fitting curve will not be returned. Return will be fit values.
        (default = False)
    return_list: boolean
        If True, fit values will be a list of the best parameter values found from the fit.
        If False, fit values will be a dictionary of the best fit parameters and their values.
        (default = False)

    Returns
    -------
    res_soliton : dict or list or tuple
        Output depends on the argument values to the function. By default the function returns a dictionary of the best
        parameters found.
    fit_sp_1d : ndarray
        If return_fit_curve is set to True the output will return a tuple of values, with the first being the curve and
        the second dependent on the setting of return_list.
    excitation_list : list
        If return_list is set to True then the output parameters will be a list of sub list, with each sub list giving
        the best parameter values for that excitation.

    """
    if inti_pos is None:
        msg = "No soliton positions provided."
        raise ValueError(msg)

    search_range = 6

    if func == "gaussian1D":
        sol_pos_1d_model = Model(gaussian1d)
        for peak in range(n):
            if peak == 0:
                sol_pos_1d_model = Model(gaussian1d, prefix=f"s{peak}_")
            else:
                sol_pos_1d_model += Model(gaussian1d, prefix=f"s{peak}_")
        sol_pos_1d_model += Model(offset_func)
        pars_soliton = sol_pos_1d_model.make_params()

        for peak in range(n):
            pars_soliton[f"s{peak}_amp"].set(value=logamp(roixwithoutbackg, inti_pos[peak]), vary=True, min=-30, max=10)
            pars_soliton[f"s{peak}_cen"].set(value=inti_pos[peak], vary=True,
                                                     min=inti_pos[peak] - search_range / 2,
                                                     max=inti_pos[peak] + search_range / 2)
            pars_soliton[f"s{peak}_sigma"].set(value=4.0, vary=True)
        pars_soliton["offset"].set(value=0.0, vary=True)
        fit_sp_1d = sol_pos_1d_model.fit(roixwithoutbackg, params=pars_soliton, x=vec_x, nan_policy="raise")
        res_soliton = fit_sp_1d.best_values

    elif func == "original":
        sol_pos_1d_model = Model(offset_func)
        for peak in range(n - 1, -1, -1):
            sol_pos_1d_model += Model(m_hat1d_old, prefix=f"s{peak}_")
        pars_soliton = Parameters()
        for peak in range(n):
            pars_soliton.add(f"s{peak}_amp", value=logamp(roixwithoutbackg, inti_pos[peak]), vary=True, min=-30, max=10)
            pars_soliton.add(f"s{peak}_cen", value=inti_pos[peak], vary=True,
                             min=inti_pos[peak] - search_range / 2, max=inti_pos[peak] + search_range / 2)
            pars_soliton.add(f"s{peak}_sigma", value=4.0, vary=True)
            pars_soliton.add(f"s{peak}_a", value=0.2, vary=True, min=-30, max=10)
            pars_soliton.add(f"s{peak}_b", value=0.0, vary=True)
        pars_soliton.add("offset", value=0.0, vary=True)
        fit_sp_1d = sol_pos_1d_model.fit(roixwithoutbackg, params=pars_soliton, x=vec_x, nan_policy="raise")
        res_soliton = fit_sp_1d.best_values

    elif func == "modern":
        sol_pos_1d_model = Model(offset_func)
        for peak in range(n - 1, -1, -1):
            sol_pos_1d_model += Model(m_hat, prefix=f"s{peak}_")
        pars_soliton = Parameters()

        for peak in range(n):
            pars_soliton.add(f"s{peak}_amp", value=logamp(roixwithoutbackg, inti_pos[peak]), vary=True, min=-30, max=10)
            pars_soliton.add(f"s{peak}_cen", value=inti_pos[peak], vary=True,
                             min=inti_pos[peak] - search_range / 2, max=inti_pos[peak] + search_range / 2)
            pars_soliton.add(f"s{peak}_sigma", value=4.0, vary=True)
            pars_soliton.add(f"s{peak}_a", value=0.2, vary=True, min=-30, max=10)
            pars_soliton.add(f"s{peak}_b", value=0.0, vary=True)
            pars_soliton.add(f"s{peak}_r0", value=res["rx"], vary=False)
            pars_soliton.add(f"s{peak}_i0", value=res["center"], vary=False)
        pars_soliton.add("offset", value=0.0, vary=True)
        fit_sp_1d = sol_pos_1d_model.fit(roixwithoutbackg, params=pars_soliton, x=vec_x, nan_policy="raise")
        res_soliton = fit_sp_1d.best_values
        for peak in range(n):
            del res_soliton[f"s{peak}_r0"]
            del res_soliton[f"s{peak}_i0"]

    else:
        msg = "No valid function provided for fitting."
        raise ValueError(msg)

    for peak in range(n):
        res_soliton[f"s{peak}_sigma"] = np.abs(res_soliton[f"s{peak}_sigma"])

    if return_list:
        excitation_list = []
        value_list = list(res_soliton.values())
        # Get total amount of params. Should be total params = (number of peaks * number of function params) + offset
        # Then subtract off the offset, which will get added to each excitation list later
        param_num = int((len(value_list) - 1) / n)
        for peak in range(n):
            excitation_list.append(value_list[peak * param_num:(peak + 1) * param_num])
            excitation_list[peak] += [value_list[-1]]

        if return_fit_curve:
            return fit_sp_1d.eval(), excitation_list
        return excitation_list

    if return_fit_curve:
        return fit_sp_1d.eval(), res_soliton
    return res_soliton


def logamp(roixwithoutbackg: np.ndarray, pos: float | None = None) -> float | int:
    """Get the log of the excitation amplitude.

    Applies a logarithm to the negative of the input, y = log(-x).

    Parameters
    ----------
    roixwithoutbackg : ndarray
        A numpy array of image data, representing the region of interest.
    pos : float
        The position of the excitation. If none is provided then the absolute minimum in the array is used.
        (default = None)

    Returns
    -------
    amp : float
        The result of the logarithm. If the value is less than -1 the output is clipped to -1.

    """
    if pos is None:
        pos = np.argmin(roixwithoutbackg)

    pos = int(pos)

    return np.log(-roixwithoutbackg[pos]) if roixwithoutbackg[pos] < -np.exp(-1) else -1


def offset_func(x: float, offset: float) -> float | np.ndarray:
    """Add an offset to a fitting function.

    Support function for fit_soliton. Serves as an offset for building a fitting model. Such as b in y = mx + b.

    Parameters
    ----------
    x : float
        Function input. Unused.
    offset : float
        The offset parameter value during fitting.

    Returns
    -------
    offset : float
        The offset parameter value during fitting.

    """
    return (x * 0) + offset


def gaussian1d(x: float, amp: float, cen: float, sigma: float) -> float | np.ndarray:
    """Fit to a 1D gaussian.

    Support function for fit_soliton. A basic gaussian fitting function.

    Parameters
    ----------
    x : float
        Function input.
    amp : float
        The amplitude of the excitation parameter during fitting.
    cen : float
        The center of the excitation parameter during fitting.
    sigma : float
        The excitation width parameter during fitting.

    Returns
    -------
    F(x) : float
        The output of the function at x.

    """
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))


def m_hat1d_old(x: float, amp: float, cen: float, sigma: float, a: float, b: float) -> float | np.ndarray:
    """Fit to a Ricker wavelet.

    Support function for fit_soliton.
    A basic Ricker wavelet fitting function used in the SolDet 1.0 library.

    Parameters
    ----------
    x : float
        Function input.
    amp : float
        The amplitude of the excitation parameter during fitting.
    cen : float
        The center of the excitation parameter during fitting.
    sigma : float
        The excitation width parameter during fitting.
    a : float
        The excitation symmetric shoulder height.
    b : float
        The excitation asymmetric shoulder height.

    Returns
    -------
    F(x) : float
        The output of the function at x.

    """
    return (-np.exp(amp) * (1 - ((cen - 82) / 82)**2)**2 * (1 - np.exp(a) * ((x - cen) / sigma)**2
            + b * ((x - cen) / sigma)) * np.exp(-(x - cen)**2 / (2 * sigma**2)))


def m_hat(x: float, amp: float, cen: float, sigma: float, a: float, b: float,
          r0: float, i0: float) -> float | np.ndarray:
    """Fit to a Ricker wavelet (modern).

    Support function for fit_soliton.
    A modified Ricker wavelet fitting function used in the modern SolDet module.

    Parameters
    ----------
    x : float
        Function input.
    amp : float
        The amplitude of the excitation parameter during fitting.
    cen : float
        The center of the excitation parameter during fitting.
    sigma : float
        The excitation width parameter during fitting.
    a : float
        The excitation symmetric shoulder height.
    b : float
        The excitation asymmetric shoulder height.
    r0 : float
        Thomas Fermi envelope width.
    i0 : float
        Thomas Fermi envelope center.

    Returns
    -------
    F(x) : float
        The output of the function at x.

    """
    return (-np.exp(amp) * (1 - ((cen - i0) / r0)**2)**2 * (1 - np.exp(a) * ((x - cen) / sigma)**2
            + b * ((x - cen) / sigma)) * np.exp(-(x - cen)**2 / (2 * sigma**2)))


def find_soliton(preprocessed_data: np.ndarray, positions: list | int | None = None, func: str = "gaussian1D",
                 *, return_list: bool = False) -> dict | list:
    """Find solitons in data.

    Return the soliton excitation fitting parameters of all excitations in the given image.
    If position is None, return the deepest depletion fitting params.

    Parameters
    ----------
    preprocessed_data : ndarray, with shape (132, 164)
        A preprocessed image.
    positions: list or float or int or None
        The horizontal location of the solitonic excitation.
        (default = None)
    func: string
        The fitting function to be used. Can be 'gaussian1D', 'original' (SOLDET 1.0), or 'modern' (SOLDET 2.0 module).
        (default = 'gaussian1D')
    return_list: boolean
        If True, fit values will be a list of the best parameter values found from the fit.
        If False, fit values will be a dictionary of the best fit parameters and their values.
        (default = False)

    Returns
    -------
    soliton_info : dict or list
        Soliton fitting parameters. See "fit_soliton" function for more information.

    """
    data = np.squeeze(preprocessed_data)
    if data.shape == (132, 164):
        vec_x, _, roixwithoutbackg, res = fit_tf_1d_from_image(data)
        if positions is None:
            positions = [np.argmin(roixwithoutbackg)]
            soliton_info = fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=positions, n=len(positions), func=func,
                                       return_list=return_list)
        elif type(positions) in {int, float, np.float64, np.float32}:
            positions = [positions]
            soliton_info = fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=positions, n=len(positions), func=func,
                                       return_list=return_list)
        elif type(positions) is list:
            soliton_info = fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=positions, n=len(positions), func=func,
                                       return_list=return_list)
        return soliton_info

    if len(data.shape) == 3 and data.shape[1:] == (132, 164):
        soliton_info = []
        if positions is None:
            for d in tqdm(data):
                vec_x, _, roixwithoutbackg, res = fit_tf_1d_from_image(d)
                positions = [np.argmin(roixwithoutbackg)]
                soliton_info.append(fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=positions, n=len(positions),
                                                func=func, return_list=return_list))
        else:
            for i, d in enumerate(tqdm(data)):
                pos = positions[i]
                vec_x, _, roixwithoutbackg, res = fit_tf_1d_from_image(d)
                if type(pos) in {int, float, np.float64, np.float32}:
                    pos = [pos]
                    soliton_info_per_image = fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=pos, n=len(pos),
                                                         func=func, return_list=return_list)
                elif type(pos) is list:
                    soliton_info_per_image = fit_soliton(vec_x, roixwithoutbackg, res, inti_pos=pos, n=len(pos),
                                                         func=func, return_list=return_list)

                soliton_info.append(soliton_info_per_image)
        return soliton_info

    msg = f"Unspported shape {preprocessed_data.shape}."
    raise ValueError(msg)


def fit_tf_1d_from_image(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit an image to a 1D Thomas Fermi model.

    Thomas fermi 1D fitting for background removal.
    This fits a 1D envelope to the image and removes it from the image to subtract the background.

    Parameters
    ----------
    roi : ndarray
        The image data to fit to.

    Returns
    -------
    vec_x : ndarray of int
        An array of the pixel positions
    vec_y : ndarray of float
        The summed array values along the height axis of the original image
    roixwithoutbackg : ndarray
        Data with the fitted background removed
    res : dict
        The fit results

    """
    vec_x = np.arange(164)
    vec_y = roi.reshape((132, 164)).sum(0)
    peaksx, peaksposx = _pickpeak(vec_y, 5)

    guess_amp = (np.mean(peaksx) - np.mean(vec_y))
    guess_cen = np.mean(peaksposx)
    guess_sigma = sum(vec_y * (vec_x - guess_cen)**2) / sum(vec_y)
    guess_sigma = np.sqrt(guess_sigma / 2)
    guess_offset = np.mean(vec_y)
    res = _fit_tf_1d(vec_x, vec_y, guess_amp, guess_cen, guess_sigma, guess_offset)
    roixwithoutbackg = (vec_y - res["fitfunc"](vec_x))

    return vec_x, vec_y, roixwithoutbackg, res


def _fit_tf_1d(x: np.ndarray, y: np.ndarray, guess_amp: float, guess_cen: float, guess_rx: float,
               guess_offset: float) -> dict:
    """Fit to a Thomas Fermi model.

    Thomas fermi 1D fitting for background removal.
    This function supports fit_tf_1d_from_image and does the actual fitting of the 1D Thomas Fermi envelope.

    Parameters
    ----------
    x : ndarray
        The input pixel position.
    y : ndarray
        The value of the array (function) at x.
    guess_amp : float
        An initial value for the amplitude of the envelope waveform.
    guess_cen : float
        An initial guess for the center of the envelope.
    guess_rx : float
        An initial guess for the width of the envelope.
    guess_offset : float
        An initial guess for the offset value of the envelope function.

    Returns
    -------
    result : dict
        A dictionary containing the fitting results. This includes the initial value and the fitted curve.

    """
    guess = np.array([guess_amp, guess_cen, guess_rx, guess_offset])
    first_guess = thomas_fermi_1dsum(x, *guess)
    popt, pcov = curve_fit(thomas_fermi_1dsum, x, y, p0=guess)
    data_fitted_tf1d = thomas_fermi_1dsum(x, *popt)
    amp, cen, rx, offset = popt
    return {"amp": amp, "center": cen, "rx": rx, "offset": offset,
            "data_fitted_tf1D": data_fitted_tf1d,
            "fitfunc": lambda x: thomas_fermi_1dsum(x, *popt), "first_guess": first_guess,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}


def thomas_fermi_1dsum(x: np.ndarray, amp: float, cen: float, rx: float, offset: float) -> float | np.ndarray:
    """Thomas fermi 1D Model for background removal.

    This function supports _fit_tf_1d and serves as the model for fitting the data to.

    Parameters
    ----------
    x : ndarray
        The input pixel position.
    amp : float
        The amplitude of the envelope parameter during fitting.
    cen : float
        The center of the envelope parameter during fitting.
    rx : float
        The width of the envelope parameter during fitting.
    offset : float
        The offset of the envelope parameter during fitting.

    Returns
    -------
    F(x) : float
        The output of the function at x

    """
    b = (1 - ((x - cen) / rx)**2)
    np.maximum(b, 0, b)

    return amp * (b**2) + offset


def outlier_treatment(datacolumn: list | np.ndarray) -> tuple[float, float]:
    """Find IQR of a given data distribution.

    Parameters
    ----------
    datacolumn : 1D list or numpy array of float numbers
        The given data distribution.

    Returns
    -------
    lower_range : float
        Lower bound of IQR
    upper_range : float
        Upper bound of IQR

    """
    sorted(datacolumn)
    q1, q3 = np.percentile(datacolumn, [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)

    return lower_range, upper_range


def preprocess_mhat_params(params: list | np.ndarray, *, remove_outliers: bool = False,
                           use_minimum_as_center: bool = False) -> np.ndarray:
    """Preprocesses the fit parameters.

    Removes the offset and optionally removes outliers or sets the minimum value as the center.

    Parameters
    ----------
    params : list or ndarray
        The fitting parameters for an excitation.
    remove_outliers: boolean
        If True, apply IQR outliers removal.
        (default = False)
    use_minimum_as_center: boolean
        If True, set the center as the minimum value.
        (default = False)

    Returns
    -------
    p : 1D or 2D ndarray
        The preprocessed parameters.

    """
    p = np.array(params)
    if len(p.shape) == 2:
        if use_minimum_as_center:
            for i, pi in enumerate(np.array(params)):
                p[i, 1] = fmin(m_hat, pi[1], tuple(pi[:-1]), disp=False)[0]
        p = p[:, :-1]  # Remove Offset

        if remove_outliers:
            for i in range(p.shape[1]):  # Remove outliers
                lower, upper = outlier_treatment(p[:, i])
                p = p[(p[:, i] < upper) & (p[:, i] > lower), :]

    elif len(p.shape) == 1:
        if use_minimum_as_center:
            p1 = fmin(m_hat, p[1], tuple(p[:-1]), disp=False)[0]
            p[1] = p1
        p = p[:-1]  # Remove Offset

    return p
