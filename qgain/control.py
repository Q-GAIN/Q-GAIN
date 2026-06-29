"""Base controller class for Q-GAIN."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class Control:
    """A base control class for Q-GAIN.

    This contains basic methods to keep a list of analysis tools. New controllers are expected to subclass this class to
    build off of.

    """

    def __init__(self, call_fn: Callable) -> None:
        """Initialize the class.

        Parameters
        ----------
        call_fn : Callable
            The function to run when the controller object is called.

        """
        self.tools = []
        self.call_fn = call_fn

    def __call__(self, tool_list: list | None = None, tool_path: list[Path] | None = None, **kwargs: dict) -> None:
        """Call this function when Controller object used as a function."""
        alerted = False
        for tool in self.tools:
            if tool_list is None or tool["name"] in tool_list:
                if not alerted:
                    print(f"Starting {self.__class__.__name__}.")
                    alerted = True
                task = tool["name"]
                task_path = None
                if tool_path is not None:
                    for path in tool_path:
                        if tool["name"] in path.stem:
                            task_path = path
                self.call_fn(tool=task, tool_path=task_path, **kwargs)

    def add_new_tool(self, tool: type, name: str, kwargs: dict | None = None) -> None:
        """Add a new tool to the controller.

        Parameters
        ----------
        tool : Class
            Some class containing the analysis algorithm. This will return an instance of the class to the controller.
        name : str
            The name of the tool. This is used to keep track of its location.
        kwargs : dict or None
            An optional dictionary of arguments to pass to the initialization of the tool class.

        """
        if kwargs is None:
            self.tools += [{"name": name, "tool": tool()}]
        else:
            self.tools += [{"name": name, "tool": tool(**kwargs)}]

    def get_id(self, name: str) -> int | None:
        """Return the index for the specified tool.

        Parameters
        ----------
        name : string
            The name of the model being searched for.

        Returns
        -------
        idx : int
            The position of the model in the tools list.

        """
        idx = None
        for i, tool in enumerate(self.tools):
            if name == tool["name"]:
                idx = i
        return idx

    def get_tool(self, name: str) -> int | None:
        """Return the specified tool.

        Parameters
        ----------
        name : string
            The name of the tool being searched for.

        Returns
        -------
        tool : Callable
            The requested tool object.

        """
        idx = self.get_id(name=name)
        return self.tools[idx]["tool"]
