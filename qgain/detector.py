"""The base Detector class for all of Q-GAIN."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import datetime
import pickle

from qgain.io import load_data, process_data
from qgain.run_ml import MLControl
from qgain.run_plot import PlotControl, cl_plotter, od_plotter
from qgain.run_stat import StatControl
from qgain.utilities import config

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgain.control import Control

    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import Dataset


class Detector:
    """The main interface to the usage of the Q-GAIN library.

    On import, or when creating a Detector object, Q-GAIN will check to confirm if a configuration file,
    CONFIG.ini, exists in its package path. If not this is created with default values.

    These set up the target directories for the required folder structure. The data_path points to the
    directory all experimental data folders will reside in. An experiment can be specified with def_exp_name, which
    will set the target directory for where Q-GAIN's data will be saved to.

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

    Similarly, new statistical based analysis tools can be added by calling the stat controller's add_new_tool
    method. See documentation on the StatControl class for more information.

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
    stat_tools : list of dicts
        A list of dictionaries specifying the statistical based analysis methods to use after the completion of the
        ML models. This dictionary should have a 'name' key whose value is the name of the method used to call it in
        'use_models' and 'define_stat' and a 'tool' key whose value is a callable class with fit() and transform()
        methods.
        (default = None)
    stats_kwargs : list of dicts
        A list of dictionaries specifying any arguments needed to initialize the corresponding tool.
        This should match the order of that found in the stat_tools argument.
        (default = None)

    """

    def __init__(self, process_fn: Callable = process_data, *,
                 od_model: Module = None, od_dataset_fn: Dataset = None, od_loss_fn: Module = None,
                 od_aug: bool | None = None, od_kwargs: dict | None = None,
                 cl_model: Module = None, cl_dataset_fn: Dataset = None, cl_loss_fn: Module = None,
                 cl_aug: bool | None = None, cl_kwargs: dict | None = None,
                 stat_tools: list[dict] | None = None, stats_kwargs: list[dict] | None = None) -> None:
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
        stat_tools : list of dicts
            A list of dictionaries specifying the statistical based analysis methods to use after the completion of the
            ML models. This dictionary should have a 'name' key whose value is the name of the method used to call it in
            'use_models' and 'define_stat' and a 'tool' key whose value is a callable class with fit() and transform()
            methods.
            (default = None)
        stats_kwargs : list of dicts
            A list of dictionaries specifying any arguments needed to initialize the corresponding tool class object.
            This should match the order of that found in the stat_tools argument.
            (default = None)

        """
        self.exp_path, self.exp_name = config()
        self.process_fn = process_fn
        self.data = []
        self.controllers = {}
        self.add_controller(name="ML Controller", controller=MLControl())
        self.add_controller(name="Stat Controller", controller=StatControl())
        self.add_controller(name="Plot Controller", controller=PlotControl(exp_path=self.exp_path))

        if cl_model is not None:
            cl_kwargs = {} if cl_kwargs is None else cl_kwargs
            self.controllers["ML Controller"].add_new_tool(model=cl_model, name="CL", dataset_fn=cl_dataset_fn,
                                                           loss_fn=cl_loss_fn, augment=cl_aug, kwargs=cl_kwargs)
            self.controllers["Plot Controller"].add_new_tool(plot_tools=[{"name": "CL", "tool": cl_plotter}])

        if od_model is not None:
            od_kwargs = {} if od_kwargs is None else od_kwargs
            self.controllers["ML Controller"].add_new_tool(model=od_model, name="OD", dataset_fn=od_dataset_fn,
                                                           loss_fn=od_loss_fn, augment=od_aug, kwargs=od_kwargs)
            self.controllers["Plot Controller"].add_new_tool(plot_tools=[{"name": "OD", "tool": od_plotter}])

        if stat_tools is not None:
            stats_kwargs = {} if stats_kwargs is None else stats_kwargs
            self.controllers["Stat Controller"].add_new_tool(stat_tools=stat_tools, stats_kwargs=stats_kwargs)

    def load_data(self, tags: list, data_frac: float = 0.9, minmax: list | None = None, *, scale: bool = False,
                  keep: bool = True) -> None:
        """Load the data corresponding to the given labels in the data roster to the Detector.

        Parameters
        ----------
        tags : list
            The type of data to load. Labels specified here will load all files in the corresponding folder.
        data_frac : float
            The fraction of the data to use for training.
            (default = 0.9)
        scale : boolean
            If True the data will be scaled so it is bounded between 0 and 1.
            (default = False)
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
            self.data += load_data(self.exp_path, tags, minmax=minmax, scale=scale)
        else:
            self.data = load_data(self.exp_path, tags, minmax=minmax, scale=scale)

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
        for use in Q-GAIN. Processing functions should return a list like object of dictionaries with at minimum the
        following keys for each data point:

            - 'tag' : A descriptor of the data. This is used by Q-GAIN to determine which data to load.
            - 'data' : Target measurement data
            - 'sub_dir' : The directory a sample should reside in.

        Additional metadata keys in the dictionary will be saved.

        The keys 'tag' and 'path' (derived from 'sub_dir') will be saved to the roster file. These will also be saved as
        attributes to the data point's HDF5 file. The 'data' entry will be saved as a separate data set in the
        sample's HDF5 file. Any other keys will be saved as attributes to the HDF5 file. If these happen to be a
        dictionary these will be saved as an empty dataset whose attributes are the dictionary items.

        The key 'sub_dir' determines what sub directories are created in the data path. The Q-GAIN library will attempt
        to create this structure in whatever the current experiment path is set to.

        If the key 'filename' is found then HDF data point files are saved with that naming scheme. Otherwise the files
        will be saved prepended with the current experiment name.

        Parameters
        ----------
        path : string
            The path to the folder containing the new data.
        kwargs : dictionary
            Additional arguments that can be passed to the processing function.

        Example
        -------
        .. code-block:: python

            args = {'target': 'MOT', 'atoms_name': 'atoms', 'bg_name': 'bckgrnd', 'probe_name': 'probe', 'tag': 'new'}
            qd.import_data(path='../MOTSet/', **args)
            qd.load_data(tags=["new"], data_frac=0.9, minmax=[0, 1])

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
            sample_name = self.exp_name + f"_{i}" if "filename" not in sample else Path(sample["filename"]).stem
            with h5py.File(roster_path, mode) as h5_file:
                ds = h5_file.create_dataset(sample_name, data=h5py.Empty("f"), dtype="f", shape=None)
                ds.attrs["tag"] = sample["tag"]
                ds.attrs["path"] = str(Path(sample["sub_dir"]).joinpath(sample_name + ".h5"))

            data_path = self.exp_path.joinpath("data/data_files/", sample["sub_dir"])
            if not data_path.is_dir():
                data_path.mkdir(parents=True)

            with h5py.File(data_path.joinpath(f"{sample_name}.h5"), "w") as h5_file:

                if isinstance(sample["data"], dict):
                    ds = h5_file.create_dataset("data", data=h5py.Empty("f"), dtype="f", shape=None)
                    ds.attrs["_dtype"] = "dict"
                    for subkey in sample["data"]:
                        ds.attrs[str(subkey)] = sample["data"][subkey]
                elif isinstance(sample["data"], list):
                    ds = h5_file.create_dataset("data", data=sample["data"], compression="gzip", compression_opts=6)
                    ds.attrs["_dtype"] = "list"
                elif isinstance(sample["data"], (str, int, float, bool)):
                    ds = h5_file.create_dataset("data", data=sample["data"])
                else:
                    ds = h5_file.create_dataset("data", data=sample["data"], compression="gzip", compression_opts=6)

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
                 checkpoints: list | None = None, data: list | dict | None = None) -> None:
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
        checkpoints: list
            If set to a list of saved weight files this will initialize the models with these weights before training.
            (default = None)
        data : list or dict
            The target data to train off of. By default this will be the data loaded into the detector object. It is
            split into a training and testing subset based on the value of the detector's data_frac attribute.
            (default = None)

        """
        target_data = self.data if data is None else data
        if model_list is None:
            tasks = []
            if self.controllers["ML Controller"].get_id(name="CL") is not None:
                tasks += ["classifier"]
            if self.controllers["ML Controller"].get_id(name="OD") is not None:
                tasks += ["object detector"]
            for tool in self.controllers["ML Controller"].tools:
                if tool["name"] != "CL" and tool["name"] != "OD":
                    tasks += [tool["name"]]
        else:
            tasks = list(model_list)

        if checkpoints is not None:
            tool_path = []
            for path in checkpoints:
                tool_path += [self.exp_path.joinpath("models", path)]
        else:
            tool_path = None

        tr_set, te_set = train_test_split(target_data, test_size=self.test, train_size=self.train)
        if "classifier" in tasks and self.controllers["ML Controller"].get_id(name="CL") is not None:
            self.controllers["ML Controller"].train(task_list=["CL"], train_data=tr_set, test_data=te_set,
                                                    optimizer_fn=optimizer_fn, model_path=self.exp_path,
                                                    batch_size=batch_size, patience=patience, epochs=epochs,
                                                    lr=lr, checkpoints=tool_path, return_res=False, save_weights=True)
            tasks.remove("classifier")

        if "object detector" in tasks and self.controllers["ML Controller"].get_id(name="OD") is not None:
            self.controllers["ML Controller"].train(task_list=["OD"], train_data=tr_set, test_data=te_set,
                                                    optimizer_fn=optimizer_fn, model_path=self.exp_path,
                                                    batch_size=batch_size, patience=patience, epochs=epochs,
                                                    lr=lr, checkpoints=tool_path, return_res=False, save_weights=True)
            tasks.remove("object detector")

        if len(tasks) > 0:
            self.controllers["ML Controller"].train(task_list=tasks, train_data=tr_set, test_data=te_set,
                                                    optimizer_fn=optimizer_fn, model_path=self.exp_path,
                                                    batch_size=batch_size, patience=patience, epochs=epochs,
                                                    lr=lr, checkpoints=tool_path, return_res=False, save_weights=True)

    def define_stat(self, tool_list: list[str] | None = None, data: list[dict] | dict | None = None,
                  *, save: bool = True) -> None:
        """Fit specified models to data.

        Define any statistical based methods by fitting them to the data loaded into the detector.
        Any saved fits are placed in the current experiment's models folder.
        For more control use the stat controller directly.

        Parameters
        ----------
        tool_list : list of strings
            The models to run. Options depend on the specified tools during initialization. If set to None Q-GAIN
            will run all available models.
            (default = None)
        data : list of dicts or dict
            The target data to fit to. By default this will be the data loaded into the detector object when set to
            None. If this argument is provided a value the fitting will occur on that data instead.
            (default = None)
        save : bool
            This determines whether to save the fitted tools to disk in the experiment's models folder.
            (Default = False)

        """
        self.controllers["Stat Controller"].build(self.data if data is None else data, tool_list,
                                                model_path=self.exp_path, save_state=save)

    def use_models(self, model_paths: list, model_list: list | tuple | None = None,
                   data: list | dict | None = None) -> None:
        """Use any specified models available in Q-GAIN.

        Specifying any of the options 'classifier', 'object detector', or any other names of ML tools in the argument
        *model_list* will make the function use those features. The argument *model_paths* can be used to dictate the
        trained model files in the models folder of the experiment path. Results are saved in the dictionary for each
        sample.

        For the ML models the results are saved to the dictionary for each sample under a key with the name of the ML
        tool + '_pred'.
        For the statistical methods the results are saved to the dictionary for each sample under a key whose name is
        the tool's name + '_pred'.

        Parameters
        ----------
        model_list : list or None
            The models to run. You can choose from the following:

                - 'classifier': Run the ML classifier model on the object's data.
                - 'object detector': Run the ML object detector on the object's data. This will determine the location
                  of any excitations found.
                - Any other loaded ML model. These should match the name field given to the ML tool during detector
                  initialization.
                - Any loaded statistical tools. These should match the name field given for the tool during
                  detector initialization.
                - None: If set to None Q-GAIN will run all available models.

            (default = None)
        model_paths : list
            The names of the saved weights or model parameters. These should end with the name of the tool and have the
            extension 'pt' for ML tools and 'pkl' for statistic tools.
            (default = [])
        data : list or dict
            The external data to use the models on. By default this will be the data loaded into the detector object. It
            is possible to use external data by providing a list of dicts or dicts of dicts to this argument.
            (default = None)

        """
        target_data = self.data if data is None else data

        controller_list = ["ML Controller", "Stat Controller"]
        for controller in self.controllers:
            if controller not in {"ML Controller", "Stat Controller", "Plot Controller"}:
                controller_list += [controller]

        if model_list is not None:
            tool_list = []
            for model in model_list:
                if model == "classifier":
                    tool_list += ["CL"]
                elif model == "object detector":
                    tool_list += ["OD"]
                else:
                    tool_list += [model]
        else:
            tool_list = None

        if model_paths is not None:
            tool_path = []
            for path in model_paths:
                tool_path += [self.exp_path.joinpath("models", path)]
        else:
            tool_path = None

        for controller in controller_list:
            if len(self.controllers[controller].tools) > 0:
                self.controllers[controller](data=target_data, tool_list=tool_list, tool_path=tool_path)
                for idx, item in enumerate(target_data):
                    for tool in self.controllers[controller].tools:
                        if "res" in tool and type(tool["res"][idx]) is dict:
                            for key, val in tool["res"][idx].items():
                                item[tool["name"] + "_" + str(key)] = val
                        elif "res" in tool and type(tool["res"][idx]) is not dict:
                            item[tool["name"] + "_pred"] = tool["res"][idx]
            for tool in self.controllers[controller].tools:
                if "res" in tool:
                    del tool["res"]

        self.data = target_data if data is None else self.data

    def plot_metrics(self, types: list | None = None, plot_kwargs: dict[dict] | None = None, *,
                     style: str | None = None, save: bool = False, data: list | dict | None = None) -> None:
        """Run various plotting routines and display the results.

        The types of plots shown depend on the entries in the list argument.

        Parameters
        ----------
        types : list
            The plotting methods to run. If None will run all available.
            (default = None)
        plot_kwargs : dict of dicts
            Optional arguments to be passed to the tool's callable function. This dictionary should contain the name of
            the plotting tool as a key with its value being the keyword dictionary to pass to the function.
            (default = None)
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
        plot_tasks = []
        kwarg_list = []

        # Build a list of plot tasks
        if len(self.controllers["Plot Controller"].tools) > 0 and types is None:
            for task in self.controllers["Plot Controller"].tools:
                plot_tasks += [task["name"]]

        elif len(self.controllers["Plot Controller"].tools) > 0 and types is not None:
            if "classifier" in types:
                plot_tasks += ["CL"]
            if "object detector" in types:
                plot_tasks += ["OD"]
            for tool in self.controllers["Plot Controller"].tools:
                if tool["name"] in types and tool["name"] != "CL" and tool["name"] != "OD":
                    plot_tasks += [tool["name"]]
        if len(self.controllers["Plot Controller"].tools) > 0:
            for tool in self.controllers["Plot Controller"].tools:
                self.controllers["Plot Controller"].set_save(tool_name=tool["name"], val=save)
                self.controllers["Plot Controller"].set_style(tool_name=tool["name"], val=style)
                if plot_kwargs is not None and tool["name"] in plot_kwargs:
                    kwarg_list += [plot_kwargs[tool["name"]]]
                else:
                    kwarg_list += [None]

        if len(plot_tasks) > 0:
            print("Starting plotting methods.")
            self.controllers["Plot Controller"].plot(data=target, tool_list=plot_tasks, kwarg_list=kwarg_list)

    def export(self, export_type: str = "csv", keys: list | None = None, data: list | dict | None = None) -> None:
        """Export the keys in the currently loaded dataset.

        Data is exported by default for keys "tag", "CL_pred", and "OD_pred" if available. Additional keys can be
        specified with the keys argument.

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
                - 'pkl': The output will be pickled as a dictionary object.
                - 'numpy': The output will be saved as a dictionary object to a Numpy .npy file.

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
            if "tag" in sample:
                export[idx]["Data Tag"] = sample["tag"]
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
            with file_path.open("wb") as f:
                pickle.dump(export, f, pickle.HIGHEST_PROTOCOL)
        elif export_type == "numpy":
            file_path = directory.joinpath("export_{}.npy".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
            np.save(file_path, export)
        else:
            msg = "Invalid value passed to type."
            raise ValueError(msg)

        print("Done!")

    def update_samples(self, data: list | dict) -> None:
        """Update existing data samples with entries from external data.

        This convenience function will work through the list of data samples and match the path key to that of the
        external data. When a match is found the internal sample is overwritten with the external data.

        Parameters
        ----------
        data: list or dict
            The external data that is used to replace internal data.

        """
        target_data = [data] if type(data) is dict else data

        for updated_sample in target_data:
            for idx in range(len(self.data)):
                if updated_sample["path"] == self.data[idx]["path"]:
                    self.data[idx] = updated_sample

    def generate_samples(self, keys: list | tuple | None = None) -> None:
        """Export internal data samples to a new dataset.

        By default this will save the following keys and its data: 'data' and 'tag'. Additional keys can be saved by
        using the keys argument. Data is saved in the experimental directory with the current data and time prepended by
        'DS'.

        Parameters
        ----------
        keys : list or tuple or None
            Additional keys to pull from each sample's dictionary in the dataset.
            (default = None)

        """
        target_data = []
        for sample in self.data:
            new_sample = {}
            new_sample["tag"] = sample["tag"]
            new_sample["sub_dir"] = str(Path(sample["path"]).parent)
            new_sample["data"] = sample["data"]
            for key in keys:
                if key not in new_sample:
                    new_sample[key] = sample[key]
            target_data += [new_sample]

        save_folder = "DS_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        roster_path = self.exp_path.joinpath(save_folder, "data", "data_info")
        if not roster_path.is_dir():
            roster_path.mkdir(parents=True)
        roster_path = roster_path.joinpath("data_roster.h5")
        if roster_path.is_file():
            mode = "a"
            with h5py.File(roster_path, mode) as h5_file:
                i = len(h5_file.keys())
        else:
            mode = "w"
            i = 0

        for sample in tqdm(target_data, desc="Writing data files.."):
            sample_name = self.exp_name + f"_{i}"
            with h5py.File(roster_path, mode) as h5_file:
                ds = h5_file.create_dataset(sample_name, data=h5py.Empty("f"), dtype="f", shape=None)
                ds.attrs["tag"] = sample["tag"]
                ds.attrs["path"] = str(Path(sample["sub_dir"]).joinpath(sample_name + ".h5"))

            data_path = self.exp_path.joinpath(save_folder, "data", "data_files", sample["sub_dir"])
            if not data_path.is_dir():
                data_path.mkdir(parents=True)

            with h5py.File(data_path.joinpath(f"{sample_name}.h5"), "w") as h5_file:

                if type(sample["data"]) in {float, int, str, bool}:
                    ds = h5_file.create_dataset("data", data=sample["data"])
                else:
                    ds = h5_file.create_dataset("data", data=sample["data"], compression="gzip", compression_opts=6)
                for key in sample:
                    if str(key) == "data":
                        pass
                    elif isinstance(sample[key], dict):
                        ds = h5_file.create_dataset(str(key), data=h5py.Empty("f"), dtype="f", shape=None)
                        for subkey in sample[key]:
                            ds.attrs[str(subkey)] = sample[key][subkey]
                    elif isinstance(sample[key], np.ndarray):
                        ds = h5_file.create_dataset(str(key), data=sample[key], compression="gzip", compression_opts=6)
                    else:
                        h5_file.attrs[str(key)] = sample[key]

            i += 1
            if mode == "w":
                mode = "a"
        del target_data

    def add_controller(self, name: str, controller: Control) -> None:
        """Add controller to detector.

        Parameters
        ----------
        name : str
            The name used for the controller.
        controller : Control
            The target control class to add to the detector.

        """
        self.controllers[name] = controller
