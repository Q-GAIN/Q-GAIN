"""The base Detector class for all of Q-GAIN."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm

from qgain.io import load_data, process_data
from qgain.run_metric import MetricControl
from qgain.run_ml import MLControl
from qgain.utilities import config

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import Dataset


class Detector:
    """The main interface to the usage of the Q-GAIN library.

    On import, or when creating a Detector object, Q-GAIN will check to confirm if a configuration file,
    CONFIG.ini, exists in its package path. If not this is created with default values.

    These set up the target directories for the required folder structure. The data_path points to the
    directory all experimental data folders will reside in. An experiment can be specified with def_exp_name, which
    will set the target directory for where Q-GAIN's class data will be saved to.

    .. code-block::

        data_path
        |- def_exp_name
            |- data
                |- data_files
                |- data_info
            |- models
        |- ...

    Multiple experiment folders can reside in the data path, and any detector objects created will reference the
    current def_exp_name. Changing to a different experiment folder requires creating another Detector object.

    Aspects of Q-GAIN can be replaced by replacing the function calls in the initialization process.
    These will override the default behavior of the library.

    Although a detector object comes with parameters for setting up an object detector and classifier, additional ML
    types can be added by simply calling the ML controller's add_new_task method. See documentation on the MLControl
    class for more information.

    Similarly, new statistical based analysis tools can be added by calling the metric controller's add_new_metric
    method. See documentation on the MetricControl class for more information.

    Parameters
    ----------
    process_fn : Callable Function
        The function to be called for preprocessing when importing new data.
        This is called when invoking import_data to add new data to an experimental folder.
    od_model : pytorch Module
        The ML model to use for object detection in Q-GAIN.
        (default = None)
    od_dataset_fn : pytorch Dataset
        The pytorch dataset class to use when handling data for the object detector.
        This should meet the requirements for use in a Pytorch dataloader.
        (default = None)
    od_loss_fn : pytorch Module
        The loss function to use during training of the object detector. This loss should be able to handle the
        target data and the output of the model.
        (default = None)
    od_aug : bool or None
        A flag to indicate whether or not to augment the provided data in the dataset function when training the
        object detector. If this value is set to True or False it is expected the specified dataset class or
        function call has an augment argument.
        (default = None)
    od_kwargs : dict
        A dictionary containing any extra function arguments needed to be passed to the object detector model
        (default = None)
    cl_model : pytorch Module
        The ML model to use for classification in Q-GAIN.
        (default = None)
    cl_dataset_fn : pytorch Dataset
        The pytorch dataset class to use when handling data for the classifier.
        This should meet the requirements for use in a Pytorch dataloader.
        (default = None)
    cl_loss_fn : pytorch Module
        The loss function to use during training of the classifier. This loss should be able to handle the target
        data and the output of the model.
        (default = None)
    cl_aug : bool or None
        A flag to indicate whether or not to augment the provided data in the dataset function when training the
        classifier. If this value is set to True or False it is expected the specified dataset class or
        function call has an augment argument.
        (default = None)
    cl_kwargs : dict
        A dictionary containing any extra function arguments needed to be passed to the classifier model
        (default = None)
    pi_metrics : list of dicts
        A list of dictionaries specifying the statistical based analysis methods to use after the completion of the ML
        models. This dictionary should have a 'name' key whose value is the name of the metric used to call it in
        'use_models' and 'define_PI' and a 'metric' key whose value is a callable class with fit() and transform()
        methods.
        (default = None)
    pi_kwargs : list of dicts
        A list of dictionaries specifying any arguments needed to initialize the corresponding metric class object.
        This should match the order of that found in the pi_metrics argument.
        (default = None)

    """

    def __init__(self, process_fn: Callable = process_data, *,
                 od_model: Module = None, od_dataset_fn: Dataset = None, od_loss_fn: Module = None,
                 od_aug: bool | None = None, od_kwargs: dict | None = None,
                 cl_model: Module = None, cl_dataset_fn: Dataset = None, cl_loss_fn: Module = None,
                 cl_aug: bool | None = None, cl_kwargs: dict | None = None,
                 pi_metrics: list[dict] | None = None, pi_kwargs: list[dict] | None = None) -> None:
        """Initialize the Detector class object.

        Parameters
        ----------
        process_fn : Callable Function
            The function to be called for preprocessing when importing new data.
            This is called when invoking import_data to add new data to an experimental folder.
        od_model : pytorch Module
            The ML model to use for object detection in Q-GAIN.
            (default = None)
        od_dataset_fn : pytorch Dataset
            The pytorch dataset class to use when handling data for the object detector.
            This should meet the requirements for use in a Pytorch dataloader.
            (default = None)
        od_loss_fn : pytorch Module
            The loss function to use during training of the object detector. This loss should be able to handle the
            target data and the output of the model.
            (default = None)
        od_aug : bool or None
            A flag to indicate whether or not to augment the provided data in the dataset function when training the
            object detector. If this value is set to True or False it is expected the specified dataset class or
            function call has an augment argument.
            (default = None)
        od_kwargs : dict
            A dictionary containing any extra function arguments needed to be passed to the object detector model
            (default = None)
        cl_model : pytorch Module
            The ML model to use for classification in Q-GAIN.
            (default = None)
        cl_dataset_fn : pytorch Dataset
            The pytorch dataset class to use when handling data for the classifier.
            This should meet the requirements for use in a Pytorch dataloader.
            (default = None)
        cl_loss_fn : pytorch Module
            The loss function to use during training of the classifier. This loss should be able to handle the target
            data and the output of the model.
            (default = None)
        cl_aug : bool or None
            A flag to indicate whether or not to augment the provided data in the dataset function when training the
            classifier. If this value is set to True or False it is expected the specified dataset class or
            function call has an augment argument.
            (default = None)
        cl_kwargs : dict
            A dictionary containing any extra function arguments needed to be passed to the classifier model
            (default = None)
        pi_metrics : list of dicts
            A list of dictionaries specifying the statistical based analysis methods to use after the completion of the
            ML models. This dictionary should have a 'name' key whose value is the name of the metric used to call it in
            'use_models' and 'define_PI' and a 'metric' key whose value is a callable class with fit() and transform()
            methods.
            (default = None)
        pi_kwargs : list of dicts
            A list of dictionaries specifying any arguments needed to initialize the corresponding metric class object.
            This should match the order of that found in the pi_metrics argument.
            (default = None)

        """
        self.data = []
        self.ml_top = MLControl()
        self.pi_top = MetricControl()

        if cl_model is not None:
            cl_kwargs = {} if cl_kwargs is None else cl_kwargs
            self.ml_top.add_new_tool(model=cl_model, name="CL", dataset_fn=cl_dataset_fn, loss_fn=cl_loss_fn,
                                     augment=cl_aug, kwargs=cl_kwargs)

        if od_model is not None:
            od_kwargs = {} if od_kwargs is None else od_kwargs
            self.ml_top.add_new_tool(model=od_model, name="OD", dataset_fn=od_dataset_fn, loss_fn=od_loss_fn,
                                     augment=od_aug, kwargs=od_kwargs)

        if pi_metrics is not None:
            pi_kwargs = {} if pi_kwargs is None else pi_kwargs
            self.pi_top.add_new_metric(pi_metrics=pi_metrics, pi_kwargs=pi_kwargs)

        self.exp_path, self.exp_name = config()
        self.process_fn = process_fn

    def __cl_plotter(self, cl_ground: list, cl_pred: list, style: str, *, save: bool) -> None:
        """Plot some metrics relevant for the classifier.

        Support function for plot_metrics.
        """
        classes = np.unique([cl_ground, cl_pred]).tolist()
        gmatrix = np.zeros((len(classes), len(classes)), dtype=int)
        g_id = 0
        p_id = 0
        for ground, pred in zip(cl_ground, cl_pred):
            for index, class_label in enumerate(classes):
                g_id = index if ground == class_label else g_id
                p_id = index if pred == class_label else p_id
            gmatrix[p_id, g_id] += 1

        if style is not None:
            with plt.style.context(style):
                fig, ax = plt.subplots()
                _ = ax.imshow(gmatrix)

                ax.set_xticks(np.arange(len(classes)), classes)
                ax.set_yticks(np.arange(len(classes)), classes)

                for i in range(len(classes)):
                    for j in range(len(classes)):
                        _ = ax.text(j, i, gmatrix[i, j], ha="center", va="center", color="w")

                ax.set_title("Dataset Labels Vs. Classifier Predictions")
                ax.set_ylabel("Classifier Predictions")
                ax.set_xlabel("Dataset Labels")
                plt.tight_layout()

                if save:
                    fig.savefig(self.exp_path.joinpath("cl_truthTable.png"))
            plt.show()
        else:
            fig, ax = plt.subplots()
            _ = ax.imshow(gmatrix)

            ax.set_xticks(np.arange(len(classes)), classes)
            ax.set_yticks(np.arange(len(classes)), classes)

            for i in range(len(classes)):
                for j in range(len(classes)):
                    _ = ax.text(j, i, gmatrix[i, j], ha="center", va="center", color="w")

            ax.set_title("Dataset Labels Vs. Classifier Predictions")
            ax.set_ylabel("Classifier Predictions")
            ax.set_xlabel("Dataset Labels")
            plt.tight_layout()

            if save:
                fig.savefig(self.exp_path.joinpath("cl_truthTable.png"))

    def __od_plotter(self, od_ground: list, od_pred: list, od_skip_count: int, style: str, *, save: bool) -> None:
        """Plot some metrics relevant for the object detector.

        Support function for plot_metrics.
        """
        if od_skip_count > 0:
            print(f"Warning: {od_skip_count} number of prediction values did not match length of OD ground labels.")

        min_val = np.min([np.min(od_ground), np.min(od_pred)])
        max_val = np.max([np.max(od_ground), np.max(od_pred)])
        bins = np.linspace(min_val, max_val, 20)
        m, b = np.polyfit(od_ground, od_pred, 1)
        x = np.array(od_ground)
        y = m * x + b
        if style is not None:
            with plt.style.context(style):
                fig, ax = plt.subplots()
                _, bins, _ = ax.hist(od_ground, bins=bins, edgecolor="black", label="Dataset Positions")
                _ = ax.hist(od_pred, bins=bins, edgecolor="black", label="Predicted Positions", alpha=0.5)

                ax.set_title("Position Histogram")
                ax.set_ylabel("Counts")
                ax.set_xlabel("Position")
                ax.tick_params(axis="both", which="major")
                ax.legend()
                plt.tight_layout()

                if save:
                    fig.savefig(self.exp_path.joinpath("od_hist.png"))

                fig, ax = plt.subplots()
                ax.set_title("Position Scatter Plot")
                ax.scatter(od_ground, od_pred, label="(Dataset Position, Predicted Position) Values", alpha=0.5)
                ax.plot(x, y, color="red", label=f"Fitted Line\nm = {m}\nb = {b}")
                ax.set_ylabel("Predicted Position")
                ax.set_xlabel("Dataset Position")
                ax.legend()
                plt.tight_layout()

                if save:
                    fig.savefig(self.exp_path.joinpath("od_scatter.png"))
            plt.show()
        else:
            bins = np.linspace(min_val, max_val, 20)
            fig, ax = plt.subplots()
            _, bins, _ = ax.hist(od_ground, bins=bins, edgecolor="black", label="Dataset Positions")
            _ = ax.hist(od_pred, bins=bins, edgecolor="black", label="Predicted Positions", alpha=0.5)

            ax.set_title("Position Histogram")
            ax.set_ylabel("Counts")
            ax.set_xlabel("Position")
            ax.tick_params(axis="both", which="major")
            ax.legend()
            plt.tight_layout()

            if save:
                fig.savefig(self.exp_path.joinpath("od_hist.png"))

            fig, ax = plt.subplots()
            ax.set_title("Position Scatter Plot")
            ax.scatter(od_ground, od_pred, label="(Dataset Position, Predicted Position) Values", alpha=0.5)
            ax.plot(x, y, color="red", label=f"Fitted Line\nm = {m}\nb = {b}")
            ax.set_ylabel("Predicted Position")
            ax.set_xlabel("Dataset Position")
            ax.legend()
            plt.tight_layout()

            if save:
                fig.savefig(self.exp_path.joinpath("od_scatter.png"))

    def __filter_ml(self, model_paths: list, model_list: list | tuple | None = None) -> tuple:
        """Seperate the ML tasks from the non-ML tasks.

        This is a support function for use_models().

        Parameters
        ----------
        model_list : list or None
            The models to run. You can choose from the following:

                - 'classifier': Run the ML classifier model on the object's data.
                - 'object detector': Run the ML object detector on the object's data. This will determine the location
                  of any excitations found.
                - Any other loaded ML model. These should match the name field given to the ML tool during detector
                  initialization.
                - Any loaded physically informed metrics. These should match the name field given for the metric during
                  detector initialization.
                - None: If set to None Q-GAIN will run all available models.

            (default = None)
        model_paths : list
            The names of the saved weights or model parameters. These should end in 'classifier.pt' for the classifier
            and 'object.pt' for the object detector. For the PI metrics these should match the filenames for any saved
            fittings.
            (default = [])

        Returns
        -------
        res : tuple
            Returns a tuple of lists with the first list being the ML tasks and the second list being the checkpoint
            paths.

        """
        weights = self.exp_path.joinpath("models")
        ml_tasks = []
        ml_files = []

        # Build a list of ML analysis
        if len(self.ml_top.tools) > 0 and model_list is None:
            for task in self.ml_top.tools:
                ml_tasks += [task["name"]]

        elif len(self.ml_top.tools) > 0 and model_list is not None:
            if "classifier" in model_list:
                ml_tasks += ["CL"]
            if "object detector" in model_list:
                ml_tasks += ["OD"]
            for task in self.ml_top.tools:
                if task["name"] in model_list and task["name"] != "CL" and task["name"] != "OD":
                    ml_tasks += [task["name"]]
        # Seperate out the ML checkpoint files
        for task in ml_tasks:
            for f in model_paths:
                if task + ".pt" in f:
                    ml_files += [weights.joinpath(f)]

        return ml_tasks, ml_files

    def __filter_non_ml(self, model_paths: list, model_list: list | tuple | None = None) -> tuple:
        """Seperate the non ML tasks from the ML tasks.

        This is a support function for use_models().

        Parameters
        ----------
        model_list : list or None
            The models to run. You can choose from the following:

                - 'classifier': Run the ML classifier model on the object's data.
                - 'object detector': Run the ML object detector on the object's data. This will determine the location
                  of any excitations found.
                - Any other loaded ML model. These should match the name field given to the ML tool during detector
                  initialization.
                - Any loaded physically informed metrics. These should match the name field given for the metric during
                  detector initialization.
                - None: If set to None Q-GAIN will run all available models.

            (default = None)
        model_paths : list
            The names of the saved weights or model parameters. These should end in 'classifier.pt' for the classifier
            and 'object.pt' for the object detector. For the PI metrics these should match the filenames for any saved
            fittings.
            (default = [])

        Returns
        -------
        res : tuple
            Returns a tuple of lists with the first list being the non ML tasks and the second list being the pickle
            object paths.

        """
        weights = self.exp_path.joinpath("models")
        pi_tasks = []
        pi_files = []
        # Build a list of non ML analysis
        if len(self.pi_top.tools) > 0 and model_list is None:
            for metric in self.pi_top.tools:
                pi_tasks += [metric["name"]]
        elif len(self.pi_top.tools) > 0 and model_list is not None:
            for metric in self.pi_top.tools:
                if metric["name"] in model_list:
                    pi_tasks += [metric["name"]]

        # Seperate out the PI pickle files
        for task in pi_tasks:
            for f in model_paths:
                if task + ".pkl" in f:
                    pi_files += [weights.joinpath(f)]

        return pi_tasks, pi_files

    def load_data(self, labels: list, data_frac: float = 0.9, minmax: list | None = None, *, scale: bool = True,
                  keep: bool = True) -> None:
        """Load the data corresponding to the given labels in the data roster to the Detector.

        Parameters
        ----------
        labels : list
            The classes to load. Labels specified here will load all files in the corresponding class folder.
        data_frac : float
            The fraction of the data to use for training.
            (default = 0.9)
        scale : boolean
            If True the data will be scaled so it is bounded between 0 and 1.
            (default = True)
        minmax : list or None
            If scale is set to True the data will be scaled given the minimum and maximum values specified in minmax.
            This expects [MIN, MAX].
            If set to None then the minimum and maximum values are found globally.
            (default = None)
        keep : bool
            If True this will keep existing data loaded into the Detector object, otherwise it will be
            overwritten.

        """
        if keep:
            self.data += load_data(self.exp_path, labels, minmax=minmax, scale=scale)
        else:
            self.data = load_data(self.exp_path, labels, minmax=minmax, scale=scale)

        self.train = data_frac
        self.test = 1 - data_frac

        if self.test > 1 or self.test < 0:
            msg = "Invalid validation dataset size."
            raise ValueError(msg)
        if self.train > 1 or self.train < 0:
            msg = "Invalid training dataset size."
            raise ValueError(msg)

    def import_data(self, path: str, **kwargs: dict[str, Any]) -> None:
        """Import new data into the class folders of the current experiment.

        This function will call whatever processing function has been set to preprocess the data and make it suitable
        for use in Q-GAIN. Processing functions should return a list of dictionaries, or a dictionary of dictionaries,
        with at minimum the following keys for each data entry:

            - 'label' : The intended class labels
            - 'filename' : Original filename of the unprocessed data
            - 'data' : Target measurement data
            - 'class_dir' : The directory a sample of a specific class should reside in.
            - 'positions' : The position of any excitations, if applicable

        Additional metadata keys in the dictionary will be saved.

        The keys 'label', 'original_file', and 'path' will be saved to the roster file. These will also be saved as
        attributes to the data sample's HDF5 file. The 'data' entry will be saved as a separate data set in the
        sample's HDF5 file. Any other keys, including 'positions', will be saved as attributes to the HDF5
        file. If these happen to be a dictionary these will be saved as an empty dataset whose attributes are the
        dictionary items.

        The key 'class_dir' determines what class folders are created in the data path. The Q-GAIN library will attempt
        to create this structure in whatever the current experiment data path is set to.

        Parameters
        ----------
        path : string
            The path to the folder containing the new data.
        kwargs : dictionary
            Additional arguments that can be passed to the processing function.

        Example
        -------
        .. code-block:: python

            args = {'target': 'xy', 'atoms_name': 'atoms', 'bg_name': 'background', 'probe_name': 'probe', 'label': 9}
            qd.import_data(path='../BEC_data_2023_0613/0001', **args)
            qd.load_data(labels= [9], data_frac = 0.9, minmax = [0, 1])

        """
        data = self.process_fn(path, **kwargs)

        roster_path = self.exp_path.joinpath("data/data_info")
        if not roster_path.is_dir():
            msg = f"{roster_path} is an invalid data_roster path."
            raise FileNotFoundError(msg)
        roster_path = roster_path.joinpath("data_roster.h5")
        if roster_path.is_file():
            mode = "a"
            with h5py.File(roster_path, mode) as h5_file:
                i = len(h5_file.keys())
        else:
            mode = "w"
            i = 0

        for sample in tqdm(data, desc="Writing data files.."):
            sample_name = self.exp_name + f"_{i}"
            with h5py.File(roster_path, mode) as h5_file:
                ds = h5_file.create_dataset(sample_name, data=h5py.Empty("f"), dtype="f", shape=None)
                ds.attrs["label"] = sample["label"]
                ds.attrs["original_file"] = sample["filename"]
                ds.attrs["path"] = str(Path(sample["class_dir"]).joinpath(sample_name + ".h5"))

            data_path = self.exp_path.joinpath("data/data_files/", sample["class_dir"])
            if not data_path.is_dir():
                data_path.mkdir(parents=True)

            with h5py.File(data_path.joinpath(f"{sample_name}.h5"), "w") as h5_file:

                ds = h5_file.create_dataset("data", data=sample["data"], compression="gzip",
                                            compression_opts=6)
                for key in sample:
                    if str(key) == "data":
                        pass
                    elif isinstance(sample[key], dict):
                        ds = h5_file.create_dataset(str(key), data=h5py.Empty("f"), dtype="f", shape=None)
                        for subkey in sample[key]:
                            ds.attrs[str(subkey)] = sample[key][subkey]
                    elif isinstance(sample[key], np.ndarray):
                        ds = h5_file.create_dataset(str(key), data=sample[key], compression="gzip",
                                        compression_opts=6)
                    else:
                        h5_file.attrs[str(key)] = sample[key]

            i += 1
            if mode == "w":
                mode = "a"

    def train_nn(self, model_list: list | tuple | None = None, batch_size: int = 32,
                 epochs: int = 30, patience: int = 30, optimizer_fn: Optimizer = Adam, lr: float = 1e-4,
                 data: list | dict | None = None) -> None:
        """Train machine learning based models.

        This will train the specified models. The resulting weights of the trained models are saved to the models
        folder of the current experiment.

        Parameters
        ----------
        model_list : list or tuple
            A list of models to train. Will train all available models if set to None.
            (default = None)
        patience : int
            How many epochs to wait with no improvement before terminating.
            (default = 30)
        epochs : int
            The number of iterations to train and test over all batches in their respective sets.
            (default = 30)
        optimizer_fn : pytorch Optimizer
            The optimizing function to use during training.
            (default = Adam)
        lr : float
            The learning rate to use in the optimizer.
            (default = 1e-4)
        batch_size : int
            The batch size to use during training.
            (default = 32)
        data : list or dict
            The target data to train off of. By default this will be the data loaded into the detector object. It is
            split into a training and testing subset based on the value of the detector's data_frac attribute.
            (default = None)

        """
        target_data = self.data if data is None else data
        if model_list is None:
            tasks = ["classifier", "object detector"]
            for tool in self.ml_top.tools:
                if tool["name"] != "CL" and tool["name"] != "OD":
                    tasks += [tool["name"]]
        else:
            tasks = list(model_list)

        tr_set, te_set = train_test_split(target_data, test_size=self.test, train_size=self.train)
        if "classifier" in tasks and self.ml_top.get_id(name="CL") is not None:
            self.ml_top.train(task_list=["CL"], train_data=tr_set, test_data=te_set, optimizer_fn=optimizer_fn,
                              model_path=self.exp_path, batch_size=batch_size,
                              patience=patience, epochs=epochs, lr=lr, return_res=False, save_weights=True)
            tasks.remove("classifier")

        if "object detector" in tasks and self.ml_top.get_id(name="OD") is not None:
            self.ml_top.train(task_list=["OD"], train_data=tr_set, test_data=te_set, optimizer_fn=optimizer_fn,
                              model_path=self.exp_path, batch_size=batch_size,
                              patience=patience, epochs=epochs, lr=lr, return_res=False, save_weights=True)
            tasks.remove("object detector")

        if len(tasks) > 0:
            self.ml_top.train(task_list=tasks, train_data=tr_set, test_data=te_set, optimizer_fn=optimizer_fn,
                              model_path=self.exp_path, batch_size=batch_size,
                              patience=patience, epochs=epochs, lr=lr, return_res=False, save_weights=True)

    def define_pi(self, metric_list: list[str] | None = None, data: list[dict] | dict | None = None,
                  *, save: bool = False) -> None:
        """Fit specified models to data.

        Define any statistical based methods by fitting them to the data loaded into the detector.
        Any saved metrics are placed in the current experiment's models folder.
        For more control use the metric controller directly.

        Parameters
        ----------
        metric_list : list of strings
            The models to run. Options depend on the specified metrics during initialization. If set to None Q-GAIN
            will run all available models.
            (default = None)
        data : list of dicts or dict
            The target data to fit to. By default this will be the data loaded into the detector object when set to
            None. If this argument is provided a value the fitting will occur on that data instead.
            (default = None)
        save : bool
            This determines whether to save the fitted metrics to disk in the experiment's models folder.
            (Default = False)

        """
        self.pi_top.build(self.data if data is None else data, metric_list, model_path=self.exp_path,
                                    save_state=save)

    def use_models(self, model_paths: list, model_list: list | tuple | None = None,
                   data: list | dict | None = None) -> None:
        """Use any specified models available in Q-GAIN.

        Specifying any of the options 'classifier', or 'object detector' in the argument model_list will make the
        function use those features. The argument model_paths can be used to dictate the trained model files in the
        models folder of the experiment path. Results are saved in the dictionary for each sample.

        For the ML models the results are saved to the dictionary for each sample under a key with the name of the ML
        tool + '_pred'.
        For the statistical methods the results are saved to the dictionary for each sample under a key whose name is
        the metric's class name + '_pred'.

        Parameters
        ----------
        model_list : list or None
            The models to run. You can choose from the following:

                - 'classifier': Run the ML classifier model on the object's data.
                - 'object detector': Run the ML object detector on the object's data. This will determine the location
                  of any excitations found.
                - Any other loaded ML model. These should match the name field given to the ML tool during detector
                  initialization.
                - Any loaded physically informed metrics. These should match the name field given for the metric during
                  detector initialization.
                - None: If set to None Q-GAIN will run all available models.

            (default = None)
        model_paths : list
            The names of the saved weights or model parameters. These should end in 'classifier.pt' for the classifier
            and 'object.pt' for the object detector. For the PI metrics these should match the filenames for any saved
            fittings.
            (default = [])
        data : list or dict
            The external data to use the models on. By default this will be the data loaded into the detector object. It
            is possible to use external data by providing a list of dicts or dicts of dicts to this argument.
            (default = None)

        """
        target_data = self.data if data is None else data

        ml_tasks, ml_files = self.__filter_ml(model_list=model_list, model_paths=model_paths)

        pi_tasks, pi_files = self.__filter_non_ml(model_list=model_list, model_paths=model_paths)

        if len(ml_tasks) > 0:
            print("Starting ML methods.")
            self.ml_top.predict(data=target_data, model_paths=ml_files, task_list=ml_tasks)

        if len(pi_tasks) > 0:
            print("Starting PI methods.")
            self.pi_top.apply(data=target_data, metric_list=pi_tasks,
                              metric_path=pi_files if len(pi_files) > 0 else None)

        for idx, item in enumerate(target_data):
            for tool in self.ml_top.tools:
                if tool["name"] in ml_tasks:
                    item[tool["name"] + "_pred"] = tool["res"][idx]

            for metric in self.pi_top.tools:
                if metric["name"] in pi_tasks:
                    item[metric["tool"].__class__.__name__ + "_pred"] = metric["res"][idx]

        # Remove the controller's copy of the results since we don't need them.
        for metric in self.pi_top.tools:
            if "res" in metric:
                del metric["res"]
        for tool in self.ml_top.tools:
            if "res" in tool:
                del tool["res"]

        if data is None:
            self.data = target_data

    def plot_metrics(self, types: list | tuple = ("classifier", "object detector"), style: str | None = None,
                     *, save: bool = False, data: list | dict | None = None) -> None:
        """Run various plotting routines and display the results.

        The types of plots shown depend on the entries in the list argument.

        Parameters
        ----------
        types : list
            Choosing a model type here will show appropriate plots for the data you'd typically expect from them.
            (default = ['classifier', 'object detector'])
        style : str
            An optional argument to specify a matplotlib style file and change the overall look of the plots.
            (default = None)
        save : bool
            An optional argument that will save the output rather than display it.
            (default = False)
        data : list or dict
            The external data to generate plots from. By default the target data is the data loaded into the detector
            object. If using a different target the function expects a similar structure to that of Q-GAIN.
            (default = None)

        """
        target = self.data if data is None else data

        cl_ground = []
        cl_pred = []
        od_ground = []
        od_pred = []
        od_skip_count = 0

        for item in target:
            if "classifier" in types and "label" in item and self.class_top is not None:
                cl_ground.append(item["label"])
                cl_pred.append(item["CL_pred"])

            if "object detector" in types and "positions" in item and self.od_top is not None:
                if len(item["positions"]) == len(item["OD_pred"]):
                    sorted_ground = np.sort(item["positions"], axis=0, kind="stable")
                    sorted_pred = np.sort(item["OD_pred"], axis=0, kind="stable")
                    for i in range(len(sorted_ground)):
                        if type(item["positions"][i]) in {list, tuple, np.ndarray}:
                            for j in range(len(item["positions"][i])):
                                od_ground.append(sorted_ground[i][j])
                                od_pred.append(sorted_pred[i][j])
                        else:
                            od_ground.append(sorted_ground[i])
                            od_pred.append(sorted_pred[i])
                else:
                    od_skip_count += 1

        if "classifier" in types and self.class_top is not None:
            self.__cl_plotter(cl_ground, cl_pred, style, save=save)

        if "object detector" in types and self.od_top is not None:
            self.__od_plotter(od_ground, od_pred, od_skip_count, style, save=save)

    def export(self, export_type: str = "csv", keys: list | None = None, data: list | dict | None = None) -> None:
        """Export the ground (if available) and predicted (if available) labels in the currently loaded dataset.

        Additional meta information can be saved by providing the relevant keys.

        Parameters
        ----------
        export_type : str
            Choosing a type here will save the data to the corresponding file format.
            You can choose from the following options:

                - 'csv': The output will be saved in table form to a csv file.
                - 'hdf': The output will be saved to a HDF format. This will allow the export of additional data types
                  such as dictionaries and arrays. Each sample will be saved to a group with its labels saved as
                  attributes. Any keys referencing dictionaries will be saved as an empty dataset in the group whose
                  entries will become attributes to the dataset. Any keys referencing arrays will be saved as datasets
                  in the group.
                - 'html': The output will be saved in table form to a html file.
                - 'pkl': The output will be pickled as a pandas dataframe object.
                - 'numpy': The output will be converted to a numpy record array and saved as a npy file.

            (default = csv)
        keys : list
            Additional keys to pull from each sample's dictionary in the dataset. This could potentially cause errors
            when attempting to export datatypes that are incompatible with the chosen output format.
            (default = None)
        data : list or dict
            The external target data to export keys from. By default the target data will be the data loaded into the
            detector object, but this overrides that.
            (default = None)

        """
        export = {}
        directory = self.exp_path
        target_data = self.data if data is None else data

        print("Exporting data..")
        for idx, sample in enumerate(target_data):
            export[idx] = {}
            export[idx]["File"] = sample["path"]
            if "label" in sample:
                export[idx]["Class Label"] = sample["label"]
            if "positions" in sample:
                export[idx]["Position"] = sample["positions"]
            if "CL_pred" in sample:
                export[idx]["CL_pred"] = sample["CL_pred"]
            if "OD_pred" in sample:
                export[idx]["OD_pred"] = sample["OD_pred"]

            if keys is not None:
                for key in keys:
                    if key in sample:
                        export[idx][key] = sample[key]

        df = pd.DataFrame.from_dict(export, "index")

        if export_type == "csv":
            file_path = directory.joinpath("export_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            df.to_csv(file_path, index=False)
        elif export_type == "hdf":
            file_path = directory.joinpath("export_{}.h5".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            with h5py.File(file_path, "w") as h5_file:
                for idx, val in export.items():
                    g = h5_file.create_group(str(idx))
                    for key in val:
                        if isinstance(val[key], dict):
                            ds = g.create_dataset(str(key), data=h5py.Empty("f"), dtype="f", shape=None)
                            for subkey in val[key]:
                                ds.attrs[str(subkey)] = val[key][subkey]
                        elif isinstance(val[key], (np.ndarray, list)):
                            ds = g.create_dataset(str(key), data=val[key], compression="gzip",
                                                  compression_opts=6)
                        else:
                            g.attrs[str(key)] = val[key]
        elif export_type == "html":
            file_path = directory.joinpath("export_{}.html".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            df.to_html(file_path, index=False)
        elif export_type == "pkl":
            file_path = directory.joinpath("export_{}.pkl".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            df.to_pickle(file_path)
        elif export_type == "numpy":
            file_path = directory.joinpath("export_{}.npy".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            rec = df.to_records(index=False)
            np.save(file_path, rec)
        else:
            msg = "Invalid value passed to type."
            raise ValueError(msg)

        print("Done!")
