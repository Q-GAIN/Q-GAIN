"""The models for the PIE classifier and Quality Estimator."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from qgain.soldet.mhat_metric import find_soliton, preprocess_mhat_params
from qgain.soldet.soliton_datasets import SolitonPIEClassDataset, SolitonQEClassDataset


class PIEClassifier:
    """Create a new metric on provided data for the physics informed classifier and apply it.

    This metric and the specified cut values will be used to categorize solitons into different types, based on how
    far they are from an 'ideal' soliton.

    Parameters
    ----------
    par0_cutoff : float
        The "top partial" cutoff value for the ratio of the amplitude fitting parameter from the top and bottom
        image cuts.
        (default = np.log(1.57))
    invpar0_cutoff : float
        The "bottom partial" cutoff value for the inverse ratio of the amplitude fitting parameter of the top and
        bottom image cuts.
        (default = -np.log(1.57))
    par4_hard_l_cutoff : float
        The "clockwise solitonic vortex" cutoff value for the asymmetric shoulder height fitting parameter
        differences from the top and bottom image cuts.
        (default = -0.53)
    par4_hard_g_cutoff : float
        The "counter clockwise solitonic vortex" cutoff value for the asymmetric shoulder height fitting parameter
        differences from the top and bottom image cuts.
        (default = 0.75)
    par4_soft_l_cutoff : float
        Involved with the 'weaker' categorization cut for the asymmetric shoulder height fitting parameter. This
        combined with par1_r_cutoff serves as the cut off values for "clockwise solitonic vortex" categorization if
        an earlier cut did not.
        (default = -0.41)
    par4_soft_g_cutoff : float
        Involved with the 'weaker' categorization cut for the asymmetric shoulder height fitting parameter. This
        combined with par1_l_cutoff serves as the cut off values for "counter clockwise solitonic vortex"
        categorization if an earlier cut did not.
        (default = 0.61)
    par1_l_cutoff : float
        Involved with the 'weaker' categorization cut for the center of the excitation fitting parameter. This
        combined with par4_soft_l_cutoff serves as the cut off values for "counter clockwise solitonic vortex"
        categorization if an earlier cut did not.
        (default = -3.0)
    par1_r_cutoff : float
        Involved with the 'weaker' categorization cut for the center of the excitation fitting parameter. This
        combined with par4_soft_l_cutoff serves as the cut off values for "clockwise solitonic vortex" categorization
        if an earlier cut did not.
        (default = 1.14)

    Further information can be found in the SolDet paper https://arxiv.org/abs/2111.04881
    func : string
        The fitting function to be used. Can be 'gaussian1D', 'original' (SOLDET 1.0),
        or 'modern' (SOLDET 2.0 module).
        (default = 'modern')
    transformer : Callable function
        The type of transform to use on the parameters.
        (default = PowerTransformer)

    """

    def __init__(self, par0_cutoff: float = np.log(1.57), invpar0_cutoff: float = -np.log(1.57),
                par4_hard_l_cutoff: float = -0.53, par4_hard_g_cutoff: float = 0.75,
                par4_soft_l_cutoff: float = -0.41, par4_soft_g_cutoff: float = 0.61,
                par1_l_cutoff: float = -3.0, par1_r_cutoff: float = 1.14,
                func: str = "modern", transformer: Callable[..., Any] = PowerTransformer) -> None:
        """Initialize the PIE classifier.

        Further information can be found in the SolDet paper https://arxiv.org/abs/2111.04881

        Parameters
        ----------
        par0_cutoff : float
            The "top partial" cutoff value for the ratio of the amplitude fitting parameter from the top and bottom
            image cuts.
            (default = np.log(1.57))
        invpar0_cutoff : float
            The "bottom partial" cutoff value for the inverse ratio of the amplitude fitting parameter of the top and
            bottom image cuts.
            (default = -np.log(1.57))
        par4_hard_l_cutoff : float
            The "clockwise solitonic vortex" cutoff value for the asymmetric shoulder height fitting parameter
            differences from the top and bottom image cuts.
            (default = -0.53)
        par4_hard_g_cutoff : float
            The "counter clockwise solitonic vortex" cutoff value for the asymmetric shoulder height fitting parameter
            differences from the top and bottom image cuts.
            (default = 0.75)
        par4_soft_l_cutoff : float
            Involved with the 'weaker' categorization cut for the asymmetric shoulder height fitting parameter. This
            combined with par1_r_cutoff serves as the cut off values for "clockwise solitonic vortex" categorization if
            an earlier cut did not.
            (default = -0.41)
        par4_soft_g_cutoff : float
            Involved with the 'weaker' categorization cut for the asymmetric shoulder height fitting parameter. This
            combined with par1_l_cutoff serves as the cut off values for "counter clockwise solitonic vortex"
            categorization if an earlier cut did not.
            (default = 0.61)
        par1_l_cutoff : float
            Involved with the 'weaker' categorization cut for the center of the excitation fitting parameter. This
            combined with par4_soft_l_cutoff serves as the cut off values for "counter clockwise solitonic vortex"
            categorization if an earlier cut did not.
            (default = -3.0)
        par1_r_cutoff : float
            Involved with the 'weaker' categorization cut for the center of the excitation fitting parameter. This
            combined with par4_soft_l_cutoff serves as the cut off values for "clockwise solitonic vortex"
            categorization if an earlier cut did not.
            (default = 1.14)
        func : string
            The fitting function to be used. Can be 'gaussian1D', 'original' (SOLDET 1.0),
            or 'modern' (SOLDET 2.0 module).
            (default = 'modern')
        transformer : Callable function
            The type of transform to use on the parameters.
            (default = PowerTransformer)

        """
        self.par0_cutoff = par0_cutoff
        self.invpar0_cutoff = invpar0_cutoff
        self.par4_hard_l_cutoff = par4_hard_l_cutoff
        self.par4_hard_g_cutoff = par4_hard_g_cutoff
        self.par4_soft_l_cutoff = par4_soft_l_cutoff
        self.par4_soft_g_cutoff = par4_soft_g_cutoff
        self.par1_l_cutoff = par1_l_cutoff
        self.par1_r_cutoff = par1_r_cutoff
        self.func = func
        self.transformer = transformer

    def __cutter(self, top_metrics: list | dict, bottom_metrics: list | dict, idx: int | None = None) -> int:
        """Perform classification cuts for PIE classifier.

        Returns
        -------
        class_return : int
            The soliton class.

        """
        if idx is None:
            diff0 = (top_metrics[1][0] - bottom_metrics[1][0])
            diff1 = (top_metrics[1][1] - bottom_metrics[1][1])
            diff4 = (top_metrics[1][4] / top_metrics[1][2] - bottom_metrics[1][4] / bottom_metrics[1][2])

        else:
            diff0 = (top_metrics[1][idx][0] - bottom_metrics[1][idx][0])
            diff1 = (top_metrics[1][idx][1] - bottom_metrics[1][idx][1])
            diff4 = (top_metrics[1][idx][4] / top_metrics[1][idx][2]
                     - bottom_metrics[1][idx][4] / bottom_metrics[1][idx][2])

        class_return = 0
        path_string = ""
        if diff0 > self.par0_cutoff:
            path_string += "A"
            path_string += "1"
            class_return = 1
        elif diff0 < self.invpar0_cutoff:
            path_string += "A"
            path_string += "2"
            class_return = 2
        # passed amplitude check
        else:
            path_string += "_"
            # strong assym check
            if diff4 < self.par4_hard_l_cutoff:
                path_string += "b"
                path_string += "3"
                class_return = 3
            elif diff4 > self.par4_hard_g_cutoff:
                path_string += "b"
                path_string += "4"
                class_return = 4
            else:
                path_string += "_"
                # pos check
                if diff1 < self.par1_l_cutoff:
                    # weak assym check
                    path_string += "icL"
                    if diff4 > self.par4_soft_g_cutoff:
                        path_string += "wb"
                        path_string += "4"
                        class_return = 4
                    else:
                        path_string += "wbF"
                        path_string += "5"
                        class_return = 5
                elif diff1 > self.par1_r_cutoff:
                    path_string += "icR"
                    # weak assym check
                    if diff4 < self.par4_soft_l_cutoff:
                        path_string += "wb"
                        path_string += "3"
                        class_return = 3
                    else:
                        path_string += "wbF"
                        path_string += "5"
                        class_return = 5
        return class_return

    def fit(self, data: list[dict] | dict) -> None:
        """Fitting algorith.

        Locates solitons in a set of images and fits them to the specified function set during class initialization.
        The resulting parameters are transformed into a new distribution and used to retrieve a covariance and mean
        value.

        Parameters
        ----------
        data : list of dicts or dict
            The data to be used in the fitting.

        """
        ds = SolitonPIEClassDataset(data)

        one_data = []
        one_pos = []
        for idx in range(len(ds)):
            one_data.append(ds[idx][0])
            one_pos.append(ds[idx][1])

        print("Building PIE metric.")
        one_soliton_params = find_soliton(one_data, positions=one_pos, func=self.func, return_list=True)
        one_soliton_params = [item for sublist in one_soliton_params for item in sublist]
        pie_fit_params = preprocess_mhat_params(one_soliton_params)

        self.PIE_pt = self.transformer()
        self.PIE_pt.fit(pie_fit_params)
        train_params_trans = self.PIE_pt.transform(pie_fit_params)
        self.PIE_cov = np.cov(train_params_trans, rowvar=0)
        self.PIE_means = np.mean(train_params_trans, axis=0)
        self.PIE_dim = len(self.PIE_means)

    def transform(self, data: list[dict] | dict) -> list:
        """Transform the data.

        Applies the previously fitted metric and defined cuts to determine the sub class of a solitonic excitation
        identified by the classifier and object detector ML models.

        Images are split into a top half and bottom half. Excitations are located and the metric is applied. From these
        parameters various cuts are done to classify the soliton into the following sub classes:

            - 0: Longitudinal soliton
            - 1: Top partial soliton
            - 2: Bottom partial soliton
            - 3: Clockwise solitonic vortex
            - 4: Counterclockwise solitonic vortex
            - 5: Canted

        These are determined by defining A, the amplitude fitting parameter ratio; db, the shoulder height difference;
        and dic, the center position difference and comparing them to various cuts. These cuts are defined as:

            - par0_cutoff: The "top partial" cutoff value for the ratio of the amplitude fitting parameter from the top
              and bottom image cuts.
            - invpar0_cutoff: The "bottom partial" cutoff value for the inverse ratio of the amplitude fitting
              parameter of the top and bottom image cuts.
            - par4_hard_l_cutoff: The "clockwise solitonic vortex" cutoff value for the asymmetric shoulder height
              fitting parameter differences from the top and bottom image cuts.
            - par4_hard_g_cutoff: The "counter clockwise solitonic vortex" cutoff value for the asymmetric shoulder
              height fitting parameter differences from the top and bottom image cuts.
            - par4_soft_l_cutoff: Involved with the 'weaker' categorization cut for the asymmetric shoulder height
              fitting parameter. This combined with par1_r_cutoff serves as the cut off values for "clockwise solitonic
              vortex" categorization if an earlier cut did not.
            - par4_soft_g_cutoff: Involved with the 'weaker' categorization cut for the asymmetric shoulder height
              fitting parameter. This combined with par1_l_cutoff serves as the cut off values for "counter clockwise
              solitonic vortex" categorization if an earlier cut did not.
            - par1_l_cutoff: Involved with the 'weaker' categorization cut for the center of the excitation fitting
              parameter. This combined with par4_soft_g_cutoff serves as the cut off values for "counter clockwise
              solitonic vortex" categorization if an earlier cut did not.
            - par1_r_cutoff: Involved with the 'weaker' categorization cut for the center of the excitation fitting
              parameter. This combined with par4_soft_l_cutoff serves as the cut off values for "clockwise solitonic
              vortex" categorization if an earlier cut did not.

        Further information can be found in the SolDet paper https://arxiv.org/abs/2111.04881

        Parameters
        ----------
        data : list of dicts or dict
            The data to be used for classifying into PIE types.

        Returns
        -------
        class_return : list
            A list of the identified sub classes for all found excitations.

        """
        res = []
        pbar = tqdm(range(len(data)), desc="PIE Classifier running..")
        warned = False
        for item in data:
            pos = None
            if "positions" in item:
                pos = item["positions"]
            elif "CL_pred" in item and item["CL_pred"] > 0 and "OD_pred" in item and len(item["OD_pred"]) > 0:
                pos = item["OD_pred"]
            elif not warned:
                tqdm.write("Warning: ML or position labels not found in an entry.")
                warned = True

            if pos is not None:
                img_data = item["data"]
                positions = pos
                bottom_mask = np.zeros_like(img_data)
                bottom_mask[:int(bottom_mask.shape[0] / 2), :] = 1
                top_mask = np.zeros_like(img_data)
                top_mask[int(top_mask.shape[0] / 2):, :] = 1

                top_prod = np.multiply(img_data, top_mask)
                bottom_prod = np.multiply(img_data, bottom_mask)
                top_metrics = find_soliton(top_prod, positions=positions, func=self.func, return_list=True)
                bottom_metrics = find_soliton(bottom_prod, positions=positions, func=self.func, return_list=True)

                work_list = [top_metrics, bottom_metrics]
                dim = np.squeeze(img_data).shape

                for i, soliton_params in enumerate(work_list):
                    if dim == (132, 164):
                        if soliton_params == []:
                            pred = []
                            process_params = []
                        else:
                            process_params = preprocess_mhat_params(soliton_params, use_minimum_as_center=False)
                            pred = apply_metric(process_params, transformer=self.PIE_pt, sigma=self.PIE_cov,
                                                        mu=self.PIE_means, return_dist=True)
                    elif dim[1:] == (132, 164):
                        pred = []
                        process_params = []
                        for params_per_image in soliton_params:
                            if params_per_image == []:
                                pred.append([])
                                process_params.append([])
                            else:
                                process_params_per_image = preprocess_mhat_params(params_per_image,
                                                                                  use_minimum_as_center=False)
                                pred.append(apply_metric(process_params_per_image, pt=self.PIE_pt, sigma=self.PIE_cov,
                                                        mu=self.PIE_means, return_dist=True))
                                process_params.append(process_params_per_image)

                    work_list[i] = [pred, process_params]

                top_metrics = work_list[0]
                bottom_metrics = work_list[1]

                if len(top_metrics[1].shape) > 1 and len(bottom_metrics[1].shape) > 1:
                    class_return = []
                    for idx in range(top_metrics[1].shape[0]):
                        class_return.append(self.__cutter(top_metrics=top_metrics,
                                                          bottom_metrics=bottom_metrics,
                                                          idx=idx))

                else:
                    class_return = self.__cutter(top_metrics=top_metrics,
                                                          bottom_metrics=bottom_metrics,
                                                          idx=None)
            else:
                class_return = None
            res += [class_return]
            pbar.update(1)
        pbar.close()

        return res


class QE:
    """Create a new metric on the data for the physics informed quality estimator.

    This metric will be used to determine how likely a soliton is a longitudinal soliton when applying the
    quality estimator. The metric is based on all existing single longitudinal solitons in the current dataset.

    Parameters
    ----------
    func : string
        The fitting function to be used. Can be 'gaussian1D', 'original' (SOLDET 1.0),
        or 'modern' (SOLDET 2.0 module).
        (default = 'modern')
    transformer : Callable function
        The type of transform to use on the parameters.
        (default = PowerTransformer)

    """

    def __init__(self, func: str = "modern", transformer: Callable[..., Any] = PowerTransformer) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        func : string
            The fitting function to be used. Can be 'gaussian1D', 'original' (SOLDET 1.0),
            or 'modern' (SOLDET 2.0 module).
            (default = 'modern')
        transformer : Callable function
            The type of transform to use on the parameters.
            (default = PowerTransformer)

        """
        self.func = func
        self.transformer = transformer

    def fit(self, data: list[dict] | dict) -> None:
        """Locate solitons in a set of images and fit them to the specified function set during class initialization.

        The resulting parameters are transformed into a new distribution and used to retrieve a covariance and mean
        value.

        Parameters
        ----------
        data : list of dicts or dict
            The data to be used in the fitting.

        """
        ds = SolitonQEClassDataset(data)

        one_data = []
        one_pos = []
        for idx in range(len(ds)):
            one_data.append(ds[idx][0])
            one_pos.append(ds[idx][1])

        print("Building QE metric.")
        one_soliton_params = find_soliton(one_data, positions=one_pos, func=self.func, return_list=True)
        one_soliton_params = [item for sublist in one_soliton_params for item in sublist]
        qe_fit_params = preprocess_mhat_params(one_soliton_params)

        self.QE_pt = self.transformer()
        self.QE_pt.fit(qe_fit_params)
        train_params_trans = self.QE_pt.transform(qe_fit_params)
        self.QE_cov = np.cov(train_params_trans, rowvar=0)
        self.QE_means = np.mean(train_params_trans, axis=0)
        self.QE_dim = len(self.QE_means)

    def transform(self, data: list[dict] | dict) -> list[float]:
        """Transform the data.

        Applies the previously built metric to determine the probability that a soliton identified by the classifier and
        object detector ML models is longitudinal.

        The score is calculated by taking the metric, transforming the input data, and calculating the mahalanobis
        distance between the metric and the transformed data. The chi squared is applied to this to calculate a score.

        Parameters
        ----------
        data : list of dicts or dict
            The data to be used for classifying into PIE types.

        Returns
        -------
        pred : list
            A list of the quality scores for all found excitations.

        """
        res = []
        pbar = tqdm(range(len(data)), desc="Quality Estimate running..")
        warned = False
        for item in data:
            pos = None
            if "positions" in item:
                pos = item["positions"]
            elif "CL_pred" in item and item["CL_pred"] > 0 and "OD_pred" in item and len(item["OD_pred"]) > 0:
                pos = item["OD_pred"]
            elif not warned:
                tqdm.write("Warning: ML or position labels not found in an entry.")
                warned = True

            if pos is not None:
                img_data = item["data"]
                positions = pos
                soliton_params = find_soliton(img_data, positions=positions, func=self.func, return_list=True)
                dim = np.squeeze(img_data).shape

                if dim == (132, 164):
                    if soliton_params == []:
                        pred = []
                        process_params = []
                    else:
                        process_params = preprocess_mhat_params(soliton_params, use_minimum_as_center=False)
                        pred = apply_metric(process_params, transformer=self.QE_pt, sigma=self.QE_cov, mu=self.QE_means,
                                            return_dist=False)
                elif dim[1:] == (132, 164):
                    pred = []
                    process_params = []
                    for params_per_image in soliton_params:
                        if params_per_image == []:
                            pred.append([])
                            process_params.append([])
                        else:
                            process_params_per_image = preprocess_mhat_params(params_per_image,
                                                                              use_minimum_as_center=False)
                            pred.append(apply_metric(process_params_per_image, transformer=self.QE_pt,
                                                     sigma=self.QE_cov, mu=self.QE_means, return_dist=False))
                            process_params.append(process_params_per_image)
            else:
                pred = None
            res += [pred]
            pbar.update(1)
        pbar.close()

        return res


def apply_metric(params: np.ndarray, sigma: np.ndarray, mu: np.ndarray, *,
                transformer: Callable[[np.ndarray, list], Any] | bool = False,
                return_dist: bool = False, gamma: bool = False) -> np.ndarray:
    """Given a transformed set, determine the mahalanobis distance or quality score.

    Parameters
    ----------
    params : numpy array
        The processed fit parameters for the excitation(s).
    sigma : numpy array
        The covariance matrix of the built metric.
    mu : numpy array
        The mean values of the built metric.
    transformer : boolean or sklearn Transformer object
        If False, parameters are not transformed.
        If a Transformer object then the input data is transformed using the metric.
        (default = False)
    return_dist: boolean
        If True, returns a list of the calculated Mahalanobis Distance(s).
        If False, returns a list of the calculated quality score(s).
        (default = False)
    gamma: boolean
        An optional argument that will apply the gamma function instead of the chi squared in the QE score.
        (default = False)

    Returns
    -------
    res : ndarray
        An array of distances or quality scores.

    """
    if len(params.shape) == 1:
        params = np.expand_dims(params, 0)

    params_trans = transformer.transform(params) if transformer is not False else params

    res = []
    for x in params_trans:
        m_dist_x = np.dot((x - mu).transpose(), np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x - mu))
        if return_dist:
            res.append(np.sqrt(m_dist_x))  # mahalanobis distance
        elif gamma:
            res.append(1 - stats.gamma.cdf(m_dist_x, len(mu)))
        else:
            res.append(1 - stats.chi2.cdf(m_dist_x, len(mu)))  # probability
    return np.array(res)
