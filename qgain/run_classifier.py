"""Classification Controller."""
from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# specify cpu or gpu device for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClassifierControl:
    """The Classifier Control class for the Q-GAIN package.

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
    metrics : list of dicts
        An optional list of loss metrics to use during validation. This argument expects a list of dictionaries with a
        key 'name' whose value gives the name of the metric and a key 'metric' whose value is a callable function that
        can be invoked by the controller to calculate a loss value. These will be listed during training and reported
        along with the built in metrics.
        (default = None)
    kwargs : dict
        A dictionary of arguments to pass to the specified model.

    Example
    -------
    .. code-block:: python

        kwargs = {'num_classes' : 3}

        cl_top = Classifier_Control(model = qgain.soldet.classifier_nn.MLST2021CNNmodern,
        dataset_fn = qgain.soldet.soliton_datasets.SolitonClassDataset,
        augment = True, device = 0, **kwargs)

    """

    def __init__(self, model: torch.nn.Module, dataset_fn: torch.utils.data.Dataset, device: int | None = None,
                 metrics: list[dict] | None = None, *, augment: bool | None = True, **kwargs: dict) -> None:
        """Initialize the controller.

        Parameters
        ----------
        model : pytorch Module
            The type of model to be used. If using a custom module the output of the model should be in the same shape
            as the target tensors used during training and validation.
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
        self.augment = augment
        self.rank = device if device is not None else 0
        self.device = f"cuda:{device}" if device is not None else DEVICE
        self.model = model(**kwargs).float().to(self.device)
        self.metrics = []
        if metrics is not None:
            self.metrics = metrics

    def train_class(self, train_data: list, test_data: list, optimizer_fn: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module, model_path: str | None = None, batch_size: int = 32, patience: int = 30,
                    epochs: int = 30, lr: float = 1e-4, *, return_res: bool = False,
                    save_weights: bool = False) -> dict | None:
        """Train the object's classifier model on the given data.

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
        loss_fn : pytorch loss Module
            The loss function to use during training.
        model_path : str
            The path to where weights should be saved to if save_weights = True.
            (default = None)
        save_weights : bool
            Whether to save the best weights or not.
            (default = False)
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
        accu : float
            The corresponding test accuracy if return_res = True. Here 'accuracy' is how many correct predictions
            there were.

        """
        train_ds = self.dataset_fn(train_data,
                                   augment=self.augment) if self.augment is not None else self.dataset_fn(train_data)
        test_ds = self.dataset_fn(test_data,
                                  augment=self.augment) if self.augment is not None else self.dataset_fn(test_data)

        print(f"Training with {len(train_ds)} samples and validating with {len(test_ds)} samples.")
        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

        if save_weights:
            save_path = Path(model_path).joinpath("models",
                                                  datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_classifier.pt")

        optimizer = optimizer_fn(self.model.parameters(), lr=lr)

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
                    loss = loss_fn(pred, target.long())

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach()

                train_metrics = {"loss": running_loss / len(train_dataloader)}

                # VALIDATION
                self.model.eval()
                test_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
                correct = 0
                for item in self.metrics:
                    item["res"] = 0
                with torch.no_grad():
                    for (te_dat, te_tar) in test_dataloader:
                        data, target = te_dat.to(self.device), te_tar.to(self.device)
                        pred = self.model(data)
                        test_loss += loss_fn(pred, target.long()).detach()
                        correct += (pred.argmax(1) == target).to(torch.float).mean().item()
                        for item in self.metrics:
                            item["res"] += item["metric"](pred, target)

                test_loss /= len(test_dataloader)
                correct /= len(test_dataloader)
                for item in self.metrics:
                    item["res"] /= len(test_dataloader)

                test_metrics = {"Test Loss": test_loss, "Accuracy": correct}
                for item in self.metrics:
                    test_metrics[item["name"]] = item["res"]

                if len(w) > 0:
                    tqdm.write("Warning: " + str(w[-1].message))

            checkpoint_dict = {
                "epoch": t,
                "train_metrics": train_metrics, "test_metrics": test_metrics,
                "model_state_dict": self.model.state_dict()}

            if t == 0:
                min_dict.update(test_metrics)
                if save_weights:
                    torch.save(checkpoint_dict, save_path)

            elif test_metrics["Test Loss"] < min_dict["Test Loss"]:
                patience_count = 0
                min_dict.update(test_metrics)
                if save_weights:
                    torch.save(checkpoint_dict, save_path)

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

    def class_predict(self, data: list | dict | np.ndarray, model_path: str) -> np.ndarray:
        """Make predictions using the object's model.

        Parameters
        ----------
        data : list or dict or ndarray
            The data to make predictions on.
        model_path : str
            The path to the saved weights for the model.

        Returns
        -------
        pos : list
            A list of all classes found for each provided image.

        """
        if type(data) is dict:
            ds = self.dataset_fn([data], augment=False) if self.augment is not None else self.dataset_fn([data])
        elif type(data) is list:
            ds = self.dataset_fn(data, augment=False) if self.augment is not None else self.dataset_fn(data)
        elif type(data) is np.ndarray:
            data_list = []
            if len(data.shape) >= 3:
                for i in range(data.shape[0]):
                    data_list.append({"data": data[i]})
            elif len(data.shape) == 2:
                data_list.append({"data": data})
            else:
                msg = "Input data incorrect shape. Expected or 2D or greater data."
                raise ValueError(msg)

            ds = self.dataset_fn(data_list, augment=False) if self.augment is not None else self.dataset_fn(data_list)
        else:
            msg = "Input data is invalid type. Epected list, dictionary, or numpy.ndarray"
            raise TypeError(msg)

        with warnings.catch_warnings(record=True) as w:
            target = []
            for idx in range(len(ds)):
                pred, _ = ds[idx]
                pred = pred.to(self.device)
                target.append(pred)

            checkpoint_dict = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.model.eval()
            print("Classifier model loaded.")
            res = []
            with torch.no_grad():
                print("Running model, please wait..")
                pbar = tqdm(range(len(target)), desc="Running..")
                for sample in target:
                    pred = self.model(sample.unsqueeze(0))
                    if len(w) > 0:
                        tqdm.write(str(w[0].message))
                        del w[0]
                    res.append(torch.argmax(pred).detach().cpu().numpy())
                    pbar.update(1)
                pbar.close()
            labels = np.asarray(res)
        print("Finished.")

        return labels
