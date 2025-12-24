"""Base controller class for Q-GAIN."""
from __future__ import annotations


class Control:
    """A base control class for Q-GAIN.

    This contains basic methods to keep a list of analysis tools. New controllers are expected to subclass this class to
    build off of.

    """

    def __init__(self) -> None:
        """Initialize the class."""
        self.tools = []

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
