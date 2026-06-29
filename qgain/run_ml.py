"""Machine Learning Controller."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import datetime
import warnings

from qgain.control import Control

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

if TYPE_CHECKING:
    import numpy as np

# specify cpu or gpu device for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLControl(Control):
    """The machine learning (ML) control class for the Q-GAIN package.

    When creating a new tool the controller will pass any initialization arguments to the MLTool which sets up
    the model. A tool to the ML controller is a seperate ML model it must run data through. This will be added as an
    entry in the member attribute list called tools.
    Adding additional ML tools can be accomplished with the listed parameters in the add_new_tool method.

    """

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__(call_fn=self.predict)

    def add_new_tool(self, model: torch.nn.Module, name: str, dataset_fn: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module, device: int | None = None, metrics: list[dict] | None = None,
                 *, augment: bool | None = True, kwargs: dict) -> None:
        """Add a new tool to the controller.

        Functionality can be modified with external modules by replacing them in the corresponding argument.

        Parameters
        ----------
        model : pytorch Module
            The type of model to be used. If using a custom module the output of the model should be in the same shape
            as the target tensors used during training and validation.
        name : string
            The name of the model. Used for checkpoint file names and GUI communication.
        dataset_fn : pytorch Dataset
            The dataset function used to provide data to the models. If using a custom function this should accept the
            shape and type of data you are providing to the framework. It should have an argument named augment
            to indicate whether or not to augment the data if augment is not set to None.
        augment : bool or None
            A flag to indicate whether or not to augment the provided data in the dataset function.
            (default = True)
        device : int
            Which device to use. An integer provided here will specify which GPU to run the model on. If none is
            provided then either GPU 0 will be selected, or the CPU will be used if no CUDA compatible device is
            detected. This also influences the output of the GUI. Worker progress will be printed to the terminal on
            lines that correspond to the device. For GPU 0 updates will be printed to position 0, for GPU 1 updates will
            be printed to position 1, and so on.
            (default = None)
        loss_fn : pytorch loss Module
            The loss function to use during training.
        metrics : list of dicts
            An optional list of loss metrics to use during validation. This argument expects a list of dictionaries with
            a key 'name' whose value gives the name of the metric and a key 'metric' whose value is a callable function
            that can be invoked by the controller to calculate a loss value. These will be listed during training and
            reported along with the built in metrics.
            (default = None)
        kwargs : dict
            A dictionary of arguments to pass to the specified model.

        """
        if metrics is not None and type(metrics) is not list:
            msg = "metrics must be a list of dictionaries."
            raise ValueError(msg)
        super().add_new_tool(name=name, tool=MLTool, kwargs={"model": model, "name": name, "dataset_fn": dataset_fn,
                                                             "loss_fn": loss_fn, "device": device, "metrics": metrics,
                                                             "augment": augment, "kwargs": kwargs})

    def train(self, train_data: list, test_data: list, optimizer_fn: torch.optim.Optimizer,
              task_list: list[str] | None = None, model_path: str | None = None, batch_size: int = 32,
              patience: int = 30, epochs: int = 30, lr: float = 1e-4, checkpoints: list | None = None,
              *, return_res: bool = False, save_weights: bool = False) -> list | None:
        """Train a tool's model.

        Specifying None in task_list trains all available models. Otherwise the controller will only train the model
        specified.

        Parameters
        ----------
        train_data : list
            The data to train the model off of. By default this expects a list of dictionaries containing the N samples.
            If a custom dataset has been specified, this data should be of the expected type and shape for that pytorch
            dataset function.
        test_data : list
            The data to test the model with. By default this expects a list of dictionaries containing the N samples.
            If a custom dataset has been specified, this data should be of the expected type and shape for that pytorch
            dataset function.
        task_list : list of strings
            The list of models to train. If None will train all available models.
            (default = None)
        optimizer_fn : pytorch Optimizer
            The optimizing function to use during training.
        model_path : str
            The path to where weights should be saved to if save_weights = True.
            (default = None)
        save_weights : bool
            Whether to save the best weights or not.
            (default = False)
        checkpoints: list
            If set to a list of saved weight files this will initialize the model with these weights before training.
            (default = None)
        batch_size : int
            The batch size to use during training.
            (default = 32)
        patience : int
            How many epochs to wait with no improvement before terminating.
            (default = 30)
        epochs : int
            The number of iterations to train and test over all batches in their respective sets.
            (default = 30)
        lr : float
            The learning rate to use in the optimizer.
            (default = 1e-4)
        return_res : bool
            Whether to return the best loss and accuracy metrics.
            (default = False)

        Returns
        -------
        results : list
            If return_res = True then function will return a list holding the results for each model. Each entry will be
            a tuple containing the minimum test loss found during training and a dictionary containing additional
            loss metrics, if provided.

        """
        # Check if all items in task list are in the controller.
        names = []
        for tool in self.tools:
            names += [tool["name"]]
        for task in task_list:
            if task not in names:
                msg = "Invalid task name given to controller."
                raise ValueError(msg)

        results = []
        for tool in self.tools:
            if task_list is None or tool["name"] in task_list:
                print("Initiating training of: {} model.".format(tool["name"]))
                checkpoint = None
                if checkpoints is not None:
                    for file in checkpoints:
                        if torch.load(file, weights_only=True)["name"] == tool["name"]:
                            checkpoint = file

                if return_res:
                    results += [tool["tool"].train(train_data=train_data, test_data=test_data,
                                                   optimizer_fn=optimizer_fn, model_path=model_path,
                                                   batch_size=batch_size, patience=patience, epochs=epochs, lr=lr,
                                                   checkpoint=checkpoint, return_res=return_res,
                                                   save_weights=save_weights)]
                else:
                    tool["tool"].train(train_data=train_data, test_data=test_data,
                                                   optimizer_fn=optimizer_fn, model_path=model_path,
                                                   batch_size=batch_size, patience=patience, epochs=epochs, lr=lr,
                                                   checkpoint=checkpoint, return_res=return_res,
                                                   save_weights=save_weights)
        if return_res:
            return results
        return None

    def predict(self, data: list | dict | np.ndarray, tool_path: Path, tool: str) -> None:
        """Make predictions using the tool's model.

        Runs all available models if task_list is set to None.

        Parameters
        ----------
        data : list or dict or ndarray
            The data to make predictions on.
        tool : str
            The ML tool to run.
        tool_path : Path object
            A Path object that points to the saved weights for the model.

        """
        for ml_tool in self.tools:
            if ml_tool["name"] == tool:
                ml_tool["res"] = ml_tool["tool"].predict(data=data, model_path=tool_path)


class MLTool:
    """A tool class intended to handle training and inference tasks for Pytorch models.

    Functionality can be modified with external modules by replacing them in the corresponding argument during class
    initialization.

    Parameters
    ----------
    model : pytorch Module
        The type of model to be used. If using a custom module the output of the model should be in the same shape as
        the target tensors used during training and validation.
    dataset_fn : pytorch Dataset
        The dataset function used to provide data to the models. If using a custom function this should accept the
        shape and type of data you are providing to the framework. It should have an argument named augment
        to indicate whether or not to augment the data if augment is not set to None.
    augment : bool or None
        A flag to indicate whether or not to augment the provided data in the dataset function.
        (default = True)
    device : int
        Which device to use. An integer provided here will specify which GPU to run the model on. If none is provided
        then either GPU 0 will be selected, or the CPU will be used if no CUDA compatible device is detected.
        This also influences the output of the GUI. Worker progress will be printed to the terminal on lines that
        correspond to the device. For GPU 0 updates will be printed to position 0, for GPU 1 updates will be printed to
        position 1, and so on.
        (default = None)
    loss_fn : pytorch loss Module
            The loss function to use during training.
    metrics : list of dicts
        An optional list of loss metrics to use during validation. This argument expects a list of dictionaries with a
        key 'name' whose value gives the name of the metric and a key 'metric' whose value is a callable function that
        can be invoked by the controller to calculate a loss value. These will be listed during training and reported
        along with the built in metrics.
        (default = None)
    kwargs : dict
        A dictionary of arguments to pass to the specified model.

    """

    def __init__(self, model: torch.nn.Module, name: str, dataset_fn: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module, device: int | None = None, metrics: list[dict] | None = None,
                 *, augment: bool | None = True, kwargs: dict | None = None) -> None:
        """Initialize the tool.

        Parameters
        ----------
        model : pytorch Module
            The type of model to be used. If using a custom module the output of the model should be in the same shape
            as the target tensors used during training and validation.
        name : str
            The name of the model. Used for checkpoint file names and GUI communication.
        dataset_fn : pytorch Dataset
            The dataset function used to provide data to the models. If using a custom function this should accept the
            shape and type of data you are providing to the framework. It should have an argument named augment
            to indicate whether or not to augment the data if augment is not set to None.
        augment : bool or None
            A flag to indicate whether or not to augment the provided data in the dataset function.
            (default = True)
        device : int
            Which device to use. An integer provided here will specify which GPU to run the model on. If none is
            provided then either GPU 0 will be selected, or the CPU will be used if no CUDA compatible device is
            detected. This also influences the output of the GUI. Worker progress will be printed to the terminal on
            lines that correspond to the device. For GPU 0 updates will be printed to position 0, for GPU 1 updates will
            be printed to position 1, and so on.
            (default = None)
        loss_fn : pytorch loss Module
            The loss function to use during training.
        metrics : list of dicts
            An optional list of loss metrics to use during validation. This argument expects a list of dictionaries with
            a key 'name' whose value gives the name of the metric and a key 'metric' whose value is a callable function
            that can be invoked by the controller to calculate a loss value. These will be listed during training and
            reported along with the built in metrics.
            (default = None)
        kwargs : dict
            A dictionary of arguments to pass to the specified model.

        """
        self.dataset_fn = dataset_fn
        self.loss_fn = loss_fn
        self.augment = augment
        self.rank = device if device is not None else 0
        self.device = f"cuda:{device}" if device is not None else DEVICE
        if kwargs is None:
            self.model = model().float()
        else:
            self.model = model(**kwargs).float()
        self.metrics = []
        self.name = name
        if metrics is not None:
            self.metrics = metrics

    def train(self, train_data: list, test_data: list, optimizer_fn: torch.optim.Optimizer,
              model_path: str | None = None, batch_size: int = 32, patience: int = 30, epochs: int = 30,
              lr: float = 1e-4, checkpoint: Path | None = None, *, return_res: bool = False,
              save_weights: bool = False) -> dict | None:
        """Train the object's model on the given data.

        Parameters
        ----------
        train_data : list
            The data to train the model off of. By default this expects a list of dictionaries containing the N samples.
            If a custom dataset has been specified, this data should be of the expected type and shape for that pytorch
            dataset function.
        test_data : list
            The data to test the model with. By default this expects a list of dictionaries containing the N samples.
            If a custom dataset has been specified, this data should be of the expected type and shape for that pytorch
            dataset function.
        optimizer_fn : pytorch Optimizer
            The optimizing function to use during training.
        model_path : str
            The path to where weights should be saved to if save_weights = True.
            (default = None)
        save_weights : bool
            Whether to save the best weights or not.
            (default = False)
        checkpoint: Path
            If set to a saved weights file this will initialize the model with these weights before training.
            (default = None)
        batch_size : int
            The batch size to use during training.
            (default = 32)
        patience : int
            How many epochs to wait with no improvement before terminating.
            (default = 30)
        epochs : int
            The number of iterations to train and test over all batches in their respective sets.
            (default = 30)
        lr : float
            The learning rate to use in the optimizer.
            (default = 1e-4)
        return_res : bool
            Whether to return the best loss and accuracy metrics.
            (default = False)

        Returns
        -------
        min_loss : float
            The minimum test loss found during training if return_res = True.
        min_dict : dict
            A dictionary containing additional loss metrics, if provided. This is returned if return_res = True.

        """
        train_ds = self.dataset_fn(train_data,
                                   augment=self.augment) if self.augment is not None else self.dataset_fn(train_data)
        test_ds = self.dataset_fn(test_data,
                                  augment=self.augment) if self.augment is not None else self.dataset_fn(test_data)

        print(f"Training {self.name} with {len(train_ds)} samples and validating with {len(test_ds)} samples.")
        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

        if save_weights:
            save_path = Path(model_path).joinpath("models",
                                                  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                                  + "_" + self.name + ".pt")

        optimizer = optimizer_fn(self.model.parameters(), lr=lr)
        get_loss = self.loss_fn()
        self.model = self.model.to(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location=torch.device(self.device),
                                                  weights_only=True)["model_state_dict"])
            print("Loaded previous checkpoint.")

        patience_count = 0
        pbar = tqdm(range(epochs),
                    desc=f"Device: {self.device} | Epoch: 0/{epochs} | Loss: #.###### | Test Loss: #.#######",
                    position=self.rank)
        min_dict = {}
        for t in pbar:
            with warnings.catch_warnings(record=True) as w:
                # TRAINING
                running_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
                self.model.train()

                for (tr_dat, tr_tar) in train_dataloader:
                    data, target = tr_dat.to(self.device), tr_tar.to(self.device)

                    # compute error
                    pred = self.model(data)
                    loss = get_loss(pred, target)

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach()

                train_metrics = {"loss": running_loss / len(train_dataloader)}

                # VALIDATION
                self.model.eval()
                test_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
                for item in self.metrics:
                    item["res"] = 0
                with torch.no_grad():
                    for (te_dat, te_tar) in test_dataloader:
                        data, target = te_dat.to(self.device), te_tar.to(self.device)
                        output = self.model(data)
                        test_loss += get_loss(output, target).detach()
                        for item in self.metrics:
                            item["res"] += item["metric"](output, target)

                test_loss /= len(test_dataloader)
                for item in self.metrics:
                    item["res"] /= len(test_dataloader)

                test_metrics = {"Test Loss": test_loss}
                for item in self.metrics:
                    test_metrics[item["name"]] = item["res"]

                if len(w) > 0:
                    tqdm.write("Warning: " + str(w[-1].message))

            if t == 0:
                min_dict.update(test_metrics)
                if save_weights:
                    torch.save({"epoch": t, "train_metrics": train_metrics, "test_metrics": test_metrics,
                                "model_state_dict": self.model.state_dict(), "name": self.name}, save_path)

            elif test_metrics["Test Loss"] < min_dict["Test Loss"]:
                patience_count = 0
                min_dict.update(test_metrics)
                if save_weights:
                    torch.save({"epoch": t, "train_metrics": train_metrics, "test_metrics": test_metrics,
                                "model_state_dict": self.model.state_dict(), "name": self.name}, save_path)

            else:
                patience_count += 1

            pbar_str = "Device: {} | Epoch: {}/{} | Loss: {:>7f} |".format(self.device, t + 1, epochs,
                                                                           train_metrics["loss"])
            for key, val in test_metrics.items():
                pbar_str += f" {key}: {val:>5f} |"
            pbar.set_description(pbar_str)

            if patience_count > patience:
                break

        pbar_str = "Done! Minimum"
        for key, val in min_dict.items():
            pbar_str += f" {key}: {val:>5f}"
        pbar_str += ".\n"
        tqdm.write(pbar_str)

        if return_res:
            return min_dict["Test Loss"].detach().cpu().item(), min_dict
        return None

    def predict(self, data: list | dict | np.ndarray, model_path: str) -> list:
        """Make predictions using the object's model.

        Parameters
        ----------
        data : list or dict or ndarray
            The data to make predictions on. This should be in the format your Dataset function supports.
        model_path : str
            The path to the saved weights for the model.

        Returns
        -------
        pos : list
            A list of all positions found for each provided image.

        """
        ds = self.dataset_fn(data, augment=False) if self.augment is not None else self.dataset_fn(data)

        with warnings.catch_warnings(record=True) as w:
            target = []
            for idx in range(len(ds)):
                x, _ = ds[idx]
                x = x.to(self.device)
                target.append(x)

            checkpoint_dict = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
            self.model = self.model.to(self.device)
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.model.eval()
            print(self.name + " model loaded.")
            res = []
            with torch.no_grad():
                print("Running model, please wait..")
                pbar = tqdm(range(len(target)), desc="Running..")
                for sample in target:
                    pred = self.model(sample.unsqueeze(0)).detach()
                    if "cpu" not in str(pred.device):
                        pred = pred.cpu()
                    pred = pred.numpy()
                    res.append(pred)
                    if len(w) > 0:
                        tqdm.write(str(w[0].message))
                        del w[0]
                    pbar.update(1)
                pbar.close()
        print("Finished.")

        return res
