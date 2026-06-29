"""The Soliton Detector Class.

This uses the Q-GAIN detector module to set up a detector suitable for detecting and analyzing solitons.

"""
from __future__ import annotations

from qgain.detector import Detector
from qgain.io import process_data
from qgain.soldet.classifier_nn import SolDetClassifier
from qgain.soldet.object_nn import MetzLoss, ObjectDetector
from qgain.soldet.pi_models import QE, PIEClassifier
from qgain.soldet.soliton_datasets import (
    SolitonClassDataset,
    SolitonODDataset,
    cl_accu_metric,
    od_accu_metric,
    pos_41labels_conversion,
)

from sklearn.preprocessing import PowerTransformer
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
import numpy as np


class SolitonDetector(Detector):
    """A Soliton Detector class to interface with the Q-GAIN library for analysis and detection of solitons.

    Parameters
    ----------
    od_kwargs : dict
        A dictionary containing any extra function arguments needed to be passed to the object detector model
        (default = {})
    cl_kwargs : dict
        A dictionary containing any extra function arguments needed to be passed to the classifier model
        (default = {"num_classes": 3})
    augment : bool or None
        A flag to indicate whether or not to augment the data in the dataset function when training ML models.
        If this value is set to True or False it is expected the specified dataset class or function call has an
        augment argument.
        (default = True)

    Example:
    -------
    .. code-block:: python

        qgain.change_exp('soldet_ds')
        qgain.soldet.soliton_datasets.download_ds()
        qgain.soldet.soliton_datasets.soldet_to_h5(soldet.config()[0])
        sd = soldet.soliton_detector.SolitonDetector()
        sd.load_data(tags = [0, 1])
        sd.train_nn(['classifier'])
        sd.use_models(model_list = ['classifier'], model_paths = ['CL.pt'])

    """

    def __init__(self, od_kwargs: dict | None = None, cl_kwargs: dict | None = None, *, augment: bool = True) -> None:
        """Initialize a detector object suitable for solitons.

        Parameters
        ----------
        od_kwargs : dict
            A dictionary containing any extra function arguments needed to be passed to the object detector model
            (default = {})
        cl_kwargs : dict
            A dictionary containing any extra function arguments needed to be passed to the classifier model
            (default = {"num_classes": 3})
        augment : bool or None
            A flag to indicate whether or not to augment the data in the dataset function when training ML models.
            If this value is set to True or False it is expected the specified dataset class or function call has an
            augment argument.
            (default = True)

        """
        od_kwargs = {} if od_kwargs is None else od_kwargs
        cl_kwargs = {"num_classes": 3} if cl_kwargs is None else cl_kwargs

        super().__init__(process_fn=process_data, od_model=ObjectDetector, od_dataset_fn=SolitonODDataset,
                         od_loss_fn=MetzLoss, cl_model=SolDetClassifier, cl_dataset_fn=SolitonClassDataset,
                         cl_loss_fn=NLLLoss, cl_aug=augment, od_aug=augment, od_kwargs=od_kwargs, cl_kwargs=cl_kwargs,
                         stat_tools=[{"name": "pie classifier", "tool": PIEClassifier},
                                       {"name": "quality estimator", "tool": QE}],
                         stats_kwargs=[{"func": "modern", "transformer": PowerTransformer},
                                      {"func": "modern", "transformer": PowerTransformer}])

        self.controllers["ML Controller"].get_tool("OD").metrics += [{"name": "Accuracy", "metric": od_accu_metric}]
        self.controllers["ML Controller"].get_tool("CL").metrics += [{"name": "Accuracy", "metric": cl_accu_metric}]

        self.controllers["Plot Controller"].add_new_tool(plot_tools=[{"name": "pie classifier",
                                                                      "tool": self.__plot_pie}])
        self.controllers["Plot Controller"].add_new_tool(plot_tools=[{"name": "quality estimator",
                                                                      "tool": self.__plot_qe}])

    @staticmethod
    def __plot_qe(data: list[dict]) -> list:
        """Plot a basic scatter plot and histogram for the quality estimates.

        This function will use the output of the QE and the SolDet dataset label to generate the plots. Like in the
        QGAIN version the resulting figures are returned so they can be saved.


        Parameters
        ----------
        data : list of dicts
                The data to generate plots from. This expects the data to be a list of dicts with the proper keys.

        Returns
        -------
        figs : list
            Returns a list containing the generated figures.

        """
        figs = []
        qe_ground = []
        qe_pred = []
        qe_total_count = 0
        qe_skip_count = 0

        for sample in data:
            if "excitation_quality" in sample:
                qe_total_count += 1
                key = "excitation_quality"
                gnd = np.array(sample[key]) if type(sample[key]) is not np.ndarray else sample[key]
                key = "quality estimator_pred"
                pred = np.array(sample[key]) if type(sample[key]) is not np.ndarray else sample[key]
                if len(pred) == len(gnd):
                    for g, p in zip(gnd, pred, strict=True):
                        qe_ground.append(g)
                        qe_pred.append(p)
                else:
                    qe_skip_count += 1

        if qe_skip_count > 0:
            print(f"Warning for QE: There were {qe_skip_count} samples skipped due to mismatched lengths."
                  f"\nThis was {100 * (qe_skip_count / len(qe_total_count)):.3f}% of the total set of PIE data.")

        min_val = np.min([np.min(qe_ground), np.min(qe_pred)])
        max_val = np.max([np.max(qe_ground), np.max(qe_pred)])
        bins = np.linspace(min_val, max_val, 20)
        m, b = np.polyfit(qe_ground, qe_pred, 1)
        x = np.array(qe_ground)
        y = m * x + b

        fig, ax = plt.subplots()
        _, bins, _ = ax.hist(qe_ground, bins=bins, edgecolor="black", label="Dataset Quality Score")
        _ = ax.hist(qe_pred, bins=bins, edgecolor="black", label="Predicted Quality Score", alpha=0.5)
        ax.set_title("Quality Estimate Histogram")
        ax.set_ylabel("Counts")
        ax.set_xlabel("Score")
        ax.tick_params(axis="both", which="major")
        ax.legend()
        plt.tight_layout()
        plt.show()

        figs += [fig]

        fig, ax = plt.subplots()
        ax.set_title("Quality Estimate Scatter Plot")
        ax.scatter(qe_ground, qe_pred, label="(Dataset QE, Predicted QE) Values", alpha=0.5)
        ax.plot(x, y, color="red", label=f"Fitted Line\nm = {m}\nb = {b}")
        ax.set_ylabel("Predicted Quality Score")
        ax.set_xlabel("Dataset Quality Score")
        ax.legend()
        plt.tight_layout()
        plt.show()

        figs += [fig]

        return figs

    @staticmethod
    def __plot_pie(data: list[dict]) -> list:
        """Plot a confusion matrix table for the PIE Classifier.

        This function will use the output of the PIE classifier and the SolDet dataset label to generate the plots. Like
        in the QGAIN version the resulting figure is returned so it can be saved.

        Parameters
        ----------
        data : list of dicts
                The data to generate plots from. This expects the data to be a list of dicts with the proper keys.

        Returns
        -------
        fig : list
            Return a list containing the generated figure.

        """
        pie_ground = []
        pie_pred = []
        pie_total_count = 0
        pie_skip_count = 0

        for sample in data:
            if "excitation_PIE" in sample:
                pie_total_count += 1
                key = "excitation_PIE"
                gnd = np.array(sample[key]) if type(sample[key]) is not np.ndarray else sample[key]
                key = "pie classifier_pred"
                pred = np.array(sample[key]) if type(sample[key]) is not np.ndarray else sample[key]
                if len(pred) == len(gnd):
                    for g, p in zip(gnd, pred, strict=True):
                        pie_ground.append(g)
                        pie_pred.append(p)
                else:
                    pie_skip_count += 1

        if pie_skip_count > 0:
            print(f"Warning for PIE Classifier: There were {pie_skip_count} samples skipped due to mismatched lengths."
                  f"\nThis was {100 * (pie_skip_count / len(pie_total_count)):.3f}% of the total set of PIE data.")

        classes = np.arange(6).tolist()
        gmatrix = np.zeros((len(classes), len(classes)), dtype=int)
        for ground, pred in zip(pie_ground, pie_pred, strict=True):
            gmatrix[int(pred), int(ground)] += 1

        fig, ax = plt.subplots()
        _ = ax.imshow(gmatrix)

        ax.set_xticks(classes)
        ax.set_yticks(classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                _ = ax.text(j, i, gmatrix[i, j], ha="center", va="center", color="w")

        ax.set_title("Dataset PIE Labels Vs. PIE Predictions")
        ax.set_ylabel("PIE Predictions")
        ax.set_xlabel("Dataset Labels")
        plt.tight_layout()
        plt.show()

        return [fig]

    def load_data(self, tags: list | tuple = (0, 1, 2, 8, 9), data_frac: float = 0.9, minmax: list | tuple = (-1, 3),
                  *, scale: bool = True, keep: bool = True) -> None:
        """Load the data corresponding to the given tags in the data roster to the SolitonDetector.

        Parameters
        ----------
        tags : list
            The classes to load. Tags specified here will load all files in the corresponding class folder.
            These labels should match the ones listed in the data.
            (default = [0, 1, 2, 8, 9])
        data_frac : float
            The fraction of the data to use for training.
            (default = 0.9)
        scale : boolean
            If True the data will be scaled so it is bounded between 0 and 1.
            (default = True)
        minmax : list
            If scale is set to True the data will be scaled given the minimum and maximum values specified in minmax.
            This expects [MIN, MAX].
            (default = [-1, 3])
        keep : bool
            If True this will keep existing data loaded into the SolitonDetector object, otherwise it will be
            overwritten.

        """
        super().load_data(tags=tags, data_frac=data_frac, minmax=minmax, scale=scale, keep=keep)

    def import_data(self, path: str, target: str = "xy", atoms_name: str = "atoms", bg_name: str = "background",
                    probe_name: str = "probe", label: int = 9, width: int = 164, height: int = 132,
                    kwargs: dict | None = None) -> None:
        """Import new data into the class folders of the current experiment.

        This function will call the Q-GAIN processing function with the original SolDet parameters to import new data
        from HDF files and preprocess the data to make it suitable for use in SolDet.
        When importing data from labscript files with different experimental configurations the parameters can be
        changed to properly search the labscript shot files.

        Parameters
        ----------
        path : string
            The path to the folder containing the new data.
        target : string
            The directory name in the h5 file containing the cloud images.
            (default = "xy")
        atoms_name : string
            The full or partial name for images containing the atoms, probe, and background.
            (default = "atoms")
        bg_name : string
            The full or partial name for images of only the background, no atoms or probe light.
            (default = "background")
        probe_name : string
            The full or partial name for images of only the probe light.
            (default = "probe")
        width : int
            The target width of the cloud images after processing.
            (default = 164)
        height : int
            The target height of the cloud images after processing.
            (default = 132)
        label : int
            The SolDet module class label for the image.
            (default = 9)
        kwargs : dict or none
            Optional arguments to pass to the processing function of the detector.
            (default = None)

        Example
        -------
        .. code-block:: python

            args = {'target': 'xy', 'atoms_name': 'atoms', 'bg_name': 'background', 'probe_name': 'probe', 'tag': 9}
            sd.import_data(path='../BEC_data_2023_0613/0001', **args)
            sd.load_data(tags=[9], data_frac=0.9, minmax=[-1, 3])

        """
        if kwargs is None:
            kwargs = {"target": target, "atoms_name": atoms_name, "bg_name": bg_name, "probe_name": probe_name,
                  "tag": label, "width": width, "height": height, "labels": {"label": label}}
        else:
            kwargs.update({"target": target, "atoms_name": atoms_name, "bg_name": bg_name, "probe_name": probe_name,
                  "tag": label, "width": width, "height": height, "labels": {"label": label}})

        super().import_data(path, **kwargs)

    def use_models(self, model_paths: list,
                   model_list: list | tuple = ("classifier", "object detector", "pie classifier", "quality estimator"),
                   data: list | dict | None = None) -> None:
        """Use all models available in the SolDet module.

        Specifying any of the options 'classifier', 'object detector', 'pie classifier', or 'quality estimator' in the
        argument model_list will make the function use those features. The argument model_paths can be used to dictate
        the trained model files in the models folder of the experiment path. If none are found then the function will
        attempt to use provided files. Results are saved in the dictionary for each sample.

        Parameters
        ----------
        model_list : list
            The models to run. You can choose from the following:

                - 'classifier': Run the ML classifier model on the object's data.
                  This will determine which class the image belongs to. For the SolDet module this is 0 (No solitons),
                  1 (Single Soliton), 2 (Multiple Solitons).

                - 'object detector': Run the ML object detector on the object's data. This will determine the location
                  of any excitations found.

                - 'pie classifier': Run the physics informed classifier on the object's data. This will further
                  classify the Solitons found by the ML models into Longitudinal, Canted, Counterclockwise Vortex,
                  Clockwise Vortex, Top Partial, and Bottom Partial.

                - 'quality estimator': Run the physics informed quality estimator. For a given soliton this will
                  estimate how much it resembles a longitudinal soliton.

            (default = ['classifier', 'object detector', 'pie classifier', 'quality estimator'])
        model_paths : list
            The names of the saved weights or model parameters. These should end in '_CL.pt' for the classifier
            and '_OD.pt' for the object detector. For conventional analysis methods these should be pickle files that
            end with the name of the method. Passing an empty list, or a list lacking any saved metric files, to this
            argument will attempt to run any defined metrics without loading files.
        data : list or dict
            The external data to use the models on. By default the target is the data loaded into a detector object. It
            is possible to use external data by providing a list of dicts or dicts of dicts to this argument.
            (default = None)

        """
        super().use_models(model_list=model_list, model_paths=model_paths, data=data)
        # By default the Q-GAIN controller just passes through the output of the model.
        # This is fine for training, but not inference. We want the human values.
        # So we convert stuff here after inference since we didn't do that in the models themselves.
        target_data = self.data if data is None else data
        for item in target_data:
            if "OD_pred" in item and type(item["OD_pred"]) is np.ndarray:
                item["OD_pred"] = pos_41labels_conversion(item["OD_pred"][0])
            if "CL_pred" in item and item["CL_pred"].flatten().shape[0] > 1:
                item["CL_pred"] = np.argmax(item["CL_pred"])
        if data is None:
            self.data = target_data

    def define_pie_classifier(self, *, save: bool = True) -> None:
        """Create a new metric on the object's data for the physics informed classifier.

        save : bool
            If true this saves the metric and cuts.
            (default = False)
        """
        super().define_stat(tool_list=["pie classifier"], save=save)

    def define_quality_estimate(self, *, save: bool = True) -> None:
        """Create a new metric on the object's data for the physics informed quality scorer.

        save : bool
            If true this saves the metric and cuts.
            (default = False)
        """
        super().define_stat(tool_list=["quality estimator"], save=save)

    def export(self, export_type: str = "csv", keys: list | None = None, data: list | dict | None = None) -> None:
        """Export the ground (if available) and predicted (if available) labels in the currently loaded dataset.

        Will export the ground and predicted labels for the object detector, classifier, quality estimator and PIE
        classifier.
        Additional meta information can be saved by providing the relevant keys.

        Parameters
        ----------
        export_type : str
            Choosing a type here will save the data to the corresponding file format.
            (default = csv)
        keys : list
            Additional keys to pull from each sample's dictionary in the dataset. This could potentially cause errors
            when attempting to export datatypes that are incompatible with the chosen output format.
            (default = None)
        data : list or dict
            The external target data to export keys from which will be used instead of the data loaded into the soliton
            detector.
            (default = None)

        """
        export_keys = ["quality estimator_pred", "pie classifier_pred"]
        if keys is not None:
            export_keys.extend(keys)
        super().export(export_type=export_type, keys=export_keys, data=data)

    def plot_metrics(self,
                     types: list | tuple = ("classifier", "object detector", "pie classifier", "quality estimator"),
                     *, style: str | None = None, save: bool = False,
                     plot_kwargs: dict[dict] | None = None, data: list | dict | None = None) -> None:
        """Run various plotting routines and display the results.

        The types of plots shown depend on the entries in the list argument.

        Parameters
        ----------
        types : list
            The plotting tools to use.
            (default = ('classifier', 'object detector', 'pie classifier', 'quality estimator'))
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
        plotter_kwargs = {"OD": {"ground_keys": ["positions"]}, "CL": {"ground_keys": ["label"]}}
        if plot_kwargs is not None:
            plotter_kwargs.update(plot_kwargs)
        super().plot_metrics(types=types, style=style, save=save, plot_kwargs=plotter_kwargs, data=data)
