"""Plotting Controller."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qgain.control import Control

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class PlotControl(Control):
    """The Plotting Control class for the Q-GAIN package."""

    def __init__(self, exp_path: str) -> None:
        """Initialize controller.

        Parameters
        ----------
        exp_path : str
            The path to where the current experiment resides.

        """
        super().__init__(call_fn=self.plot)
        self.styles = {}
        self.saves = {}
        self.path = Path(exp_path)

    def add_new_tool(self, plot_tools: list[dict], *, style: str | None = None, save: bool = False) -> None:
        """Add a new plotting method to the existing list of tools.

        This function allows a user to add additional methods after the controller has been initialized.

        Parameters
        ----------
        plot_tools : list of dicts
            A list of dictionaries specifying the plotting function to use when plot is invoked. This dictionary
            should have a 'name' key whose value is some descriptive name of the tool and a 'tool' key whose value is a
            callable function.
        style : str
            An optional argument to specify a matplotlib style file and change the overall look of the tool's output.
            (default = None)
        save : boolean
            An optional argument to specify if any generated figures are saved. The controller will look for a return
            value from the object that contains an iterable with the figures to be saved.
            (default = False)

        """
        for tool in plot_tools:
            super().add_new_tool(name=tool["name"], tool=PlotterTool, kwargs={"func": tool["tool"]})
            loc = super().get_id(name=tool["name"])
            self.styles[loc] = style
            self.saves[loc] = save

    def plot(self, data: list[dict] | dict, tool_list: list | None = None, kwarg_list: list | None = None) -> None:
        """Plot the data.

        Work through the available or specified plotting tools. This will pass a list of dictionaries
        derived from data given the specified keys.

        Parameters
        ----------
        data : list of dicts or dicts
            The data to use for the plotting methods.
        tool_list : list
            The tools to run. Options depend on the specified tools during initialization. If set to None the
            controller uses every available tool.
            (default = None)
        kwarg_list : list
            Optional arguments to be passed to the tool's calable function. This list of dicts should match the ordering
            of tool_list.
            (default = None)

        """
        if type(data) is list:
            pass
        elif type(data) is dict:
            data = [data]
        else:
            msg = f"Invalid data type {type(data)}."
            raise ValueError(msg)

        for idx, tool in enumerate(self.tools):
            if tool_list is None or tool["name"] in tool_list:
                print("Starting method: {}".format(tool["name"]))
                loc = super().get_id(name=tool["name"])
                if self.styles[loc] is not None:
                    with plt.style.context(self.styles[loc]):
                        arg_bool = False
                        if kwarg_list is not None and len(kwarg_list) > 0:
                            arg_bool = kwarg_list[idx] is not None
                        figs = tool["tool"](data, **kwarg_list[idx]) if arg_bool else tool["tool"](data)
                else:
                    with plt.style.context("default"):
                        arg_bool = False
                        if kwarg_list is not None and len(kwarg_list) > 0:
                            arg_bool = kwarg_list[idx] is not None
                        figs = tool["tool"](data, **kwarg_list[idx]) if arg_bool else tool["tool"](data)
                if figs is not None and self.saves[loc]:
                    for j, fig in enumerate(figs):
                        fig.savefig(self.path.joinpath(tool["name"] + f"_{j}.png"))
                elif figs is None and self.saves[loc]:
                    print("Warning: Asked to save figure but none returned.")

    def set_save(self, *, tool_name: str, val: bool) -> None:
        """Set whether to save the output of the tool.

        tool_name : string
            The name of the tool to set.
        val : bool
            The value to set.
        """
        loc = super().get_id(name=tool_name)
        self.saves[loc] = val

    def set_style(self, *, tool_name: str, val: str) -> None:
        """Set whether to save the output of the tool.

        tool_name : string
            The name of the tool to set.
        val : string
            The path to the style sheet.
        """
        loc = super().get_id(name=tool_name)
        self.styles[loc] = val


class PlotterTool:
    """Plotter tool for the controller.

    This serves as a tool wrapper to call a user's plotting functionality.
    """

    def __init__(self, func: Callable) -> None:
        """Initialize the plotter object.

        Parameters
        ----------
        func: Callable
            The function containing the plotting code. This is called when the class object is used as a function.

        """
        self.func = func

    def __call__(self, data: list[dict], **kwargs: dict[str, Any]) -> None:
        """Call the plotting functionality when the object is called like a function.

        data : list of dicts
            The data to generate plots from. This expects the data to be a list of dicts with the proper keys.
        kwargs: dict
            An optional dictionary of keyword arguments to pass to the function.
        """
        return self.func(data, **kwargs)


def od_plotter(data: list[dict], ground_keys: list | tuple) -> list:
    """Plot a basic scatter plot and fit a line to it.

    This function will sort and flatten the output of the object detector and whatever keys found in the ground_keys
    list. These will then be used to generate a scatter plot and fit a line to it. The resulting figure is returned so
    it can be saved.

    Parameters
    ----------
    data : list of dicts
            The data to generate plots from. This expects the data to be a list of dicts with the proper keys.
    ground_keys : list
            The ground keys in a sample's dictionary entry to plot. These will be compared against the output of the
            object detector.

    Returns
    -------
    figs : list
        Returns a list containing the generated figures.

    """
    figs = []

    for key in ground_keys:
        od_skip_count = 0
        od_total_count = 0
        od_pred = []
        od_ground = []
        for sample in data:
            if key in sample:
                od_total_count += 1
                target = np.array(sample[key]) if type(sample[key]) is not np.ndarray else sample[key]
                sorted_ground = target.flatten()
                target = np.array(sample["OD_pred"]) if type(sample["OD_pred"]) is not np.ndarray else sample["OD_pred"]
                sorted_pred = target.flatten()

                if len(sorted_pred) == len(sorted_ground):
                    for g, p in zip(sorted_ground, sorted_pred, strict=True):
                        od_pred.append(p)
                        od_ground.append(g)
                else:
                    od_skip_count += 1

        if od_skip_count > 0:
            print(f"Warning for key {key}: There were {od_skip_count} samples skipped due to mismatched lengths.")
            print(f"This was {100 * (od_skip_count / od_total_count):.3f}% of the total set of data with key '{key}'.")

        m, b = np.polyfit(od_ground, od_pred, 1)
        x = np.array(od_ground)
        y = m * x + b

        fig, ax = plt.subplots()
        ax.set_title(f" Ground {key} vs. Object Detector Output")
        ax.scatter(od_ground, od_pred, label="(Dataset Position, Predicted Position) Values", alpha=0.5)
        ax.plot(x, y, color="red", label=f"Fitted Line\nm = {m}\nb = {b}")
        ax.set_ylabel("Object Detector predictions")
        ax.set_xlabel(f"Dataset {key}")
        ax.legend()
        plt.tight_layout()
        plt.show()

        figs.append(fig)

    return figs


def cl_plotter(data: list[dict], ground_keys: list | tuple, pred_key: str = "CL_pred") -> list:
    """Plot a confusion matrix table of the data.

    This function will use the output of the classifier and whatever keys found in the ground_key list to generate the
    plots. The resulting figures are returned so they can be saved.

    Parameters
    ----------
    data : list of dicts
            The data to generate plots from. This expects the data to be a list of dicts with the proper keys.
    ground_keys : list
            The ground keys in a sample's dictionary entry to plot. These will be compared against the output of the
            classifier.
    pred_key : str
        The key indicating where the classifier model's results are.
        (default = "CL_pred")

    Returns
    -------
    figs : list
        Returns a list containing the generated figures.

    """
    figs = []

    for key in ground_keys:
        cl_ground = []
        cl_pred = []
        for sample in data:
            if key in sample:
                cl_ground.append(sample[key])
                cl_pred.append(sample[pred_key])

        classes = np.arange(np.max(np.unique([cl_ground, cl_pred])) + 1).tolist()
        gmatrix = np.zeros((len(classes), len(classes)), dtype=int)
        for ground, pred in zip(cl_ground, cl_pred, strict=True):
            gmatrix[int(pred), int(ground)] += 1

        fig, ax = plt.subplots()
        _ = ax.imshow(gmatrix)

        ax.set_xticks(np.arange(len(classes)), classes)
        ax.set_yticks(np.arange(len(classes)), classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                _ = ax.text(j, i, gmatrix[i, j], ha="center", va="center")

        ax.set_title(f" Ground {key} vs. Classifier output")
        ax.set_ylabel("Classifier predictions")
        ax.set_xlabel(f"Ground {key}")
        plt.tight_layout()
        plt.show()

        figs.append(fig)

    return figs
