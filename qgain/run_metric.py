"""Conventional Analysis Controller."""
from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class MetricControl:
    """The Metric Control class for the Q-GAIN package.

    Functionality can be added by including external modules to its metrics list. These modules should be callable
    classes which contain fit and transform methods. When the controller attempts to define a new metric it works
    through the list of provided metrics and calls the metric's corresponding fit method to generate / learn some
    parameters that are used by the transform method on new data to produce some result.

    The Transform method of any provided class is expected to return a list of results. This list is saved to the
    metric's dictionary under the key 'res'. Q-GAIN will save these results to the detector's data with the metric's
    class name as the prefix and '_pred' as the suffix.

    Parameters
    ----------
    metrics : list of dicts
        A list of dictionaries specifying the statistical based metrics to use after the completion of the ML models.
        This dictionary should have a 'name' key whose value is the name of the metric used to call it in 'use_models'
        and 'define_PI' and a 'metric' key whose value is a callable class with fit() and transform() methods.
    kwargs : list of dicts
        A list of dictionaries specifying any arguments needed to initialize the corresponding metric class object.
        This should match the order of that found in the pi_metrics argument.
        (Default = None)

    Example
    -------
    .. code-block:: python

        class dummy_PI_class():
            def __init__(self, comp: int, transformer: Callable[..., Any] = PCA):

                self.comp = comp
                self.transformer = transformer(comp)

            def fit(self, data: list[dict] | dict)
                data = []
                for item in data:
                    data += [item.flatten()]
                self.transformer.fit(np.array(data))

            def transform(self, data: list[dict] | dict):
                params = []
                for item in data:
                    params += [self.transformer.transform(np.array(item.flatten()))]
                return params

        pi_control = Metric_Control(metrics = [{'name': 'pca', 'metric': dummy_PI_class},], kwargs = [{'comp': 3},])

    """

    def __init__(self, metrics: list[dict], kwargs: list[dict] | None = None) -> None:
        """Initialize controller.

        Parameters
        ----------
        metrics : list of dicts
            A list of dictionaries specifying the statistical based metrics to use after the completion of the ML
            models. This dictionary should have a 'name' key whose value is the name of the metric used to call it in
            'use_models' and 'define_PI' and a 'metric' key whose value is a callable class with fit() and transform()
            methods.
        kwargs : list of dicts
            A list of dictionaries specifying any arguments needed to initialize the corresponding metric class object.
            This should match the order of that found in the pi_metrics argument.
            (Default = None)

        """
        self.metrics = []

        for idx, metric in enumerate(metrics):
            if kwargs is not None:
                self.metrics += [{"name": metric["name"], "metric": metric["metric"](**kwargs[idx])}]
            else:
                self.metrics += [{"name": metric["name"], "metric": metric["metric"]()}]

    def build_pi_metric(self, data: list[dict] | dict | np.ndarray, metric_list: list | None = None,
                        model_path: str | None = None, *, save_state: bool = False) -> None:
        """Fit to the data.

        For any given metric methods fit the provided data to produce some parameters. The state of the metric after
        fitting can be optionally saved to disk using the save_state argument.

        Parameters
        ----------
        data : list of dicts or dicts or ndarray
            The data to use for the fitting algorithm.

        metric_list : list
            The models to run. Options depend on the specified metrics during initialization. If set to None the
            controller will fit every available metric.
            (default = None)
        model_path : str
            The path to the folder to save any metric states.
            (default = None)
        save_state : bool
            Determines whether the controller should save the fitting to disk.
            (default = False)

        """
        save = False
        for metric in self.metrics:
            if metric_list is None or metric["name"] in metric_list:
                metric["metric"].fit(data)
                if save_state:
                    save = True

            if save:
                model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(model_path).joinpath("models", model_datetime + "_" + metric["name"] + ".pkl")
                with save_path.open("wb") as f:
                    pickle.dump(metric, f, pickle.HIGHEST_PROTOCOL)
                save = False

    def apply_pi_metric(self, data: list[dict] | dict | np.ndarray, metric_list: list | None = None,
                        metric_path: list[str] | None = None) -> None:
        """Transform a set of data given a previous fit.

        For any given metric methods apply the transform to produce some result using a previously built fitting.

        Parameters
        ----------
        data : list of dicts or dicts or ndarray
            The data to use the metric method on to produce some results.
        metric_list : list
            The models to run. Options depend on the specified metrics during initialization. If set to None the
            controller will apply every available metric.
            (default = None)
        metric_path : str
            If set to None the controller will assume the metric has been fitted to the data. Otherwise this argument
            expects a Path obhect to the pickle file holding the fitting data required for the metric to function.
            (default = None)

        """
        metric_names = []
        for metric in self.metrics:
            metric_names += [metric["name"]]

        if metric_path is not None:
            for file in metric_path:
                with file.open("rb") as f:
                    file_metric = pickle.load(f)
                    if file_metric["name"] in metric_names:
                        idx = metric_names.index(file_metric["name"])
                        self.metrics[idx] = file_metric
                    else:
                        self.metrics += [file_metric]
                    print("Loaded {}.".format(file_metric["name"]))

        for metric in self.metrics:
            if metric_list is None or metric["name"] in metric_list:
                metric["res"] = metric["metric"].transform(data)
