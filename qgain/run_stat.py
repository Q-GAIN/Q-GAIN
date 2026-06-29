"""Conventional Analysis Controller."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import datetime
import pickle

from qgain.control import Control

if TYPE_CHECKING:
    import numpy as np


class StatControl(Control):
    """The Stat Control class for the Q-GAIN package.

    Functionality can be added by including external modules to its tools list. These modules should be callable
    classes which contain fit and transform methods. When the controller attempts to define a new tool it works
    through the list of provided tools and calls the tool's corresponding fit method to generate / learn some
    parameters that are used by the transform method on new data to produce some result.

    The Transform method of any provided class is expected to return a list of results. This list is saved to the
    tool's dictionary under the key 'res'. Q-GAIN will save these results to the detector's data with the tool's
    class name as the prefix and '_pred' as the suffix.

    """

    def __init__(self) -> None:
        """Initialize controller."""
        super().__init__(call_fn=self.apply)

    def add_new_tool(self, stat_tools: list[dict], stats_kwargs: list[dict] | None = None) -> None:
        """Add a new analysis method to the existing list of tools.

        This function allows a user to add additional methods after the controller has been initialized.

        Parameters
        ----------
        stat_tools : list of dicts
            A list of dictionaries specifying the statistical based tools to use after the completion of the ML
            models. This dictionary should have a 'name' key whose value is the name of the tool used to call it in
            'use_models' and 'define_stat' and a 'tool' key whose value is a callable class with fit() and transform()
            methods.
        stats_kwargs : list of dicts
            A list of dictionaries specifying any arguments needed to initialize the corresponding tool class object.
            This should match the order of that found in the stat_tools argument.
            (Default = None)

        """
        for idx, tool in enumerate(stat_tools):
            super().add_new_tool(name=tool["name"], tool=tool["tool"], kwargs=stats_kwargs[idx])

    def build(self, data: list[dict] | dict | np.ndarray, tool_list: list | None = None,
                        model_path: str | None = None, *, save_state: bool = False) -> None:
        """Fit to the data.

        For any given tool methods fit the provided data to produce some parameters. The state of the tool after
        fitting can be optionally saved to disk using the save_state argument.

        Parameters
        ----------
        data : list of dicts or dicts or ndarray
            The data to use for the fitting algorithm.

        tool_list : list
            The models to run. Options depend on the specified tools during initialization. If set to None the
            controller will fit every available tool.
            (default = None)
        model_path : str
            The path to the folder to save any tool fitting states.
            (default = None)
        save_state : bool
            Determines whether the controller should save the fitting to disk.
            (default = False)

        """
        save = False
        for tool in self.tools:
            if tool_list is None or tool["name"] in tool_list:
                print("Starting method: {}".format(tool["name"]))
                tool["tool"].fit(data)
                if save_state:
                    save = True

            if save:
                model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(model_path).joinpath("models", model_datetime + "_" + tool["name"] + ".pkl")
                with save_path.open("wb") as f:
                    pickle.dump(tool, f, pickle.HIGHEST_PROTOCOL)
                save = False

    def apply(self, data: list[dict] | dict | np.ndarray, tool: str, tool_path: Path) -> None:
        """Transform a set of data given a previous fit.

        For any given tool methods apply the transform to produce some result using a previously built fitting.

        Parameters
        ----------
        data : list of dicts or dicts or ndarray
            The data to use the tool method on to produce some results.
        tool : str
            The model to run. Options depend on the specified tools during initialization.
        tool_path : Path object
            If set to None the controller will assume the tool has been fitted to the data. Otherwise this argument
            expects a Path object to the pickle file holding the fitting data required for the tool to function.
            (default = None)

        """
        for idx in range(len(self.tools)):
            if self.tools[idx]["name"] == tool:
                if tool_path is not None:
                    with tool_path.open("rb") as f:
                        file_tool = pickle.load(f)
                        self.tools[self.get_id(file_tool["name"])] = file_tool
                        print("Loaded {}.".format(file_tool["name"]))
                print("Starting method: {}".format(self.tools[idx]["name"]))
                self.tools[idx]["res"] = self.tools[idx]["tool"].transform(data)
