"""Unit tests for the base Q-GAIN module."""

from collections.abc import Generator
from csv import DictReader
from pathlib import Path
from typing import Any
import pickle
import shutil

from pandas import read_html
from qgain.control import Control
from qgain.detector import Detector
from qgain.run_plot import PlotterTool
from torch import Tensor, from_numpy
from torch.nn import L1Loss, Linear, Module, SmoothL1Loss
from torch.utils.data import Dataset
import h5py
import matplotlib as mpl
import numpy as np
import pytest
import qgain

mpl.use("Agg")


class TrialNN(Module):
    """A test NN model."""

    def __init__(self, test: str) -> None:
        """Initialize test model."""
        super().__init__()
        self.msg = test
        self.layer = Linear(32, 32)

    def forward(self, x: Tensor) -> Tensor:
        """Take a tensor and do a single pass.

        Returns
        -------
        Result of forward pass.

        """
        return self.layer(x)


class TrialDataset(Dataset):
    """A test dataset."""

    def __init__(self, data: list, *, augment: bool = False) -> None:
        """Initialize the class."""
        self.data = []
        for item in data:
            self.data += [from_numpy(item["data"]).float()]
        self.augment = augment

    def __len__(self) -> int:
        """Return the length.

        Returns
        -------
        length : int

        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Retrieve an element at the index.

        Parameters
        ----------
        idx : int
            Which element to retrieve.

        Returns
        -------
        data : Tensor

        """
        return self.data[idx], self.data[idx]


class TrialStat:
    """A test stat tool."""

    def __init__(self, test_param: str | None = None) -> None:
        """Initialize the class."""
        self.msg = test_param
        self.params = None

    def fit(self, x: list[dict]) -> None:
        """Fit the data."""
        data = []
        for item in x:
            data += [item["data"]]
        data = np.array(data).mean(axis=0)
        self.params = np.polynomial.polynomial.Polynomial.fit(np.arange(len(data)), np.array(data), 3).coef

    def transform(self, x: list[dict]) -> list:
        """Apply a fit to the data.

        Paramaters
        ----------
        x : list of dicts
            The data to run the test tool on.

        Returns
        -------
        result: list
            The results of the transformation.

        """
        result = []
        for item in x:
            val = 0
            for order in range(self.params.shape[0]):
                val += self.params[order] * (item["data"][0]**order)
            result += [val]
        return result


def import_func(path: None, ret_type: str) -> list:
    """Test import function.

    Parameters
    ----------
    path : None
        Does nothing.
    ret_type : string
        Determines what the import function returns.

    Returns
    -------
    result : np.ndarray or dict or list

    """
    if path is not None:
        print(path)
    if ret_type == "numpy":
        data = np.array([1, 2, 3])
    if ret_type == "list":
        data = ["Hello World.", "The answer is 42."]
    if ret_type == "dict":
        data = {"token": "Hello World", "vector": np.array([1, 2, 3])}
    if ret_type == "str":
        data = "Hello World"
    if ret_type == "val":
        data = 42
    return [{"data": data, "tag": "test", "type": ret_type, "sub_dir": "test", "test_dic": {"P1": 42},
             "test_list": ["Hi"]}]


@pytest.fixture(scope="session", autouse=True)
def setup_paths() -> Generator[Any]:
    """Configure and clean paths.

    Yields
    ------
    None

    """
    path, exp_name = qgain.config()
    yield
    qgain.change_path(str(path.parent))
    qgain.change_exp(exp_name)
    print("\nRestoring paths.")


def test_paths(tmp_path: Path) -> None:
    """Test creation of Q-GAIN paths."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    new_path, new_name = qgain.config()
    assert str(new_path.parent) == str(tmp_path)
    assert new_name == "test_exp"
    assert tmp_path.joinpath("test_exp").is_dir()
    assert tmp_path.joinpath("test_exp", "data").is_dir()
    assert tmp_path.joinpath("test_exp", "models").is_dir()


def test_creation(tmp_path: Path) -> None:
    """Test creation of Q-GAIN object."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    new_path, _ = qgain.config()

    assert qgain.Detector()
    qd = qgain.Detector()

    assert isinstance(qd, Detector)
    assert qd.exp_name == "test_exp"
    assert str(qd.exp_path) == str(new_path)
    for controller in qd.controllers:
        assert isinstance(qd.controllers[controller], Control)
    shutil.rmtree(path=str(tmp_path.joinpath("test_exp")))


def test_import(tmp_path: Path) -> None:
    """Test data import."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")

    qd = qgain.Detector(process_fn=import_func)
    qd.import_data(path=None, ret_type="numpy")
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test").is_dir()
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test", "test_exp_0.h5").is_file()

    qd.import_data(path=None, ret_type="list")
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test").is_dir()
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test", "test_exp_1.h5").is_file()

    qd.import_data(path=None, ret_type="dict")
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test").is_dir()
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test", "test_exp_2.h5").is_file()

    qd.import_data(path=None, ret_type="str")
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test").is_dir()
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test", "test_exp_3.h5").is_file()

    qd.import_data(path=None, ret_type="val")
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test").is_dir()
    assert tmp_path.joinpath("test_exp", "data", "data_files", "test", "test_exp_4.h5").is_file()


def test_load(tmp_path: Path) -> None:
    """Test data loading."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")

    qd = qgain.Detector(process_fn=import_func)
    qd.import_data(path=None, ret_type="numpy")
    qd.import_data(path=None, ret_type="list")
    qd.import_data(path=None, ret_type="dict")
    qd.import_data(path=None, ret_type="str")
    qd.import_data(path=None, ret_type="val")

    qd.load_data(tags=["test"], scale=False)

    # This should be a numpy array
    assert (qd.data[0]["data"] == np.array([1, 2, 3])).all()
    assert qd.data[0]["type"] == "numpy"
    assert qd.data[0]["path"] == "test/test_exp_0.h5"

    # This should be an array of strings
    assert qd.data[1]["data"] == ["Hello World.", "The answer is 42."]
    assert qd.data[1]["type"] == "list"
    assert qd.data[1]["path"] == "test/test_exp_1.h5"

    # This should be a dictionary
    for key, val in {"token": "Hello World", "vector": np.array([1, 2, 3])}.items():
        assert key in qd.data[2]["data"]
        if isinstance(val, np.ndarray):
            assert (val == qd.data[2]["data"][key]).all()
        else:
            assert val == qd.data[2]["data"][key]
    assert qd.data[2]["type"] == "dict"
    assert qd.data[2]["path"] == "test/test_exp_2.h5"

    # This should be a string
    assert qd.data[3]["data"] == "Hello World"
    assert qd.data[3]["type"] == "str"
    assert qd.data[3]["path"] == "test/test_exp_3.h5"

    assert qd.data[4]["data"] == 42
    assert qd.data[4]["type"] == "val"
    assert qd.data[4]["path"] == "test/test_exp_4.h5"


def test_add_controller(tmp_path: Path) -> None:
    """Test adding controllers."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")

    qd = qgain.Detector()
    qd.add_controller(name="test Controller", controller=qgain.run_ml.MLControl)
    assert qd.controllers["test Controller"]
    assert isinstance(qd.controllers["test Controller"](), qgain.run_ml.MLControl)


def test_add_nn(tmp_path: Path) -> None:
    """Test adding NN models."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for item in np.random.default_rng().random((64, 32)):
        test_data += [{"data": item}]

    # Test OD
    qd = qgain.Detector(od_model=TrialNN, od_dataset_fn=TrialDataset, od_loss_fn=L1Loss, od_aug=True,
                        od_kwargs={"test": "hello"})
    assert isinstance(qd.controllers["ML Controller"].get_tool("OD").model, TrialNN)
    assert isinstance(qd.controllers["ML Controller"].get_tool("OD").dataset_fn(test_data), TrialDataset)
    assert isinstance(qd.controllers["ML Controller"].get_tool("OD").loss_fn(), L1Loss)
    assert qd.controllers["ML Controller"].get_tool("OD").augment
    assert qd.controllers["ML Controller"].get_tool("OD").model.msg == "hello"

    # Test CL
    qd = qgain.Detector(cl_model=TrialNN, cl_dataset_fn=TrialDataset, cl_loss_fn=L1Loss, cl_aug=True,
                        cl_kwargs={"test": "hello"})
    assert isinstance(qd.controllers["ML Controller"].get_tool("CL").model, TrialNN)
    assert isinstance(qd.controllers["ML Controller"].get_tool("CL").dataset_fn(test_data), TrialDataset)
    assert isinstance(qd.controllers["ML Controller"].get_tool("CL").loss_fn(), L1Loss)
    assert qd.controllers["ML Controller"].get_tool("CL").augment
    assert qd.controllers["ML Controller"].get_tool("CL").model.msg == "hello"

    # Test add_tool
    qd = qgain.Detector()
    qd.controllers["ML Controller"].add_new_tool(model=TrialNN, name="Trial", dataset_fn=TrialDataset, loss_fn=L1Loss,
                                                 metrics=[{"name": "SmoothL1", "metric": SmoothL1Loss}],
                                                 kwargs={"test": "hello"})
    assert isinstance(qd.controllers["ML Controller"].get_tool("Trial").model, TrialNN)
    assert isinstance(qd.controllers["ML Controller"].get_tool("Trial").dataset_fn(test_data), TrialDataset)
    assert isinstance(qd.controllers["ML Controller"].get_tool("Trial").loss_fn(), L1Loss)
    assert qd.controllers["ML Controller"].get_tool("Trial").augment
    assert qd.controllers["ML Controller"].get_tool("Trial").model.msg == "hello"
    assert qd.controllers["ML Controller"].get_tool("Trial").metrics[0]["name"] == "SmoothL1"
    assert isinstance(qd.controllers["ML Controller"].get_tool("Trial").metrics[0]["metric"](), SmoothL1Loss)


def test_nn(tmp_path: Path) -> None:
    """Test NN tool usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")

    qd = qgain.Detector()
    qd.controllers["ML Controller"].add_new_tool(model=TrialNN, name="Trial", dataset_fn=TrialDataset, loss_fn=L1Loss,
                                                 metrics=[{"name": "SmoothL1", "metric": SmoothL1Loss()}],
                                                 kwargs={"test": "hello"})
    qd.train = 0.9
    qd.test = 0.1

    test_data = []
    for item in np.random.default_rng().random((64, 32)):
        test_data += [{"data": item}]

    qd.train_nn(epochs=5, data=test_data)

    file_paths = list(tmp_path.joinpath("test_exp", "models").rglob("*.pt"))
    model_checkpoints = []
    for file in file_paths:
        model_checkpoints += [file.name]

    assert len(model_checkpoints) > 0

    # Test not listing any models
    qd.use_models(model_paths=model_checkpoints, data=test_data)
    for item in test_data:
        assert "Trial_pred" in item
        del item["Trial_pred"]
        assert "Trial_pred" not in item

    # Test listing model
    qd.use_models(model_paths=model_checkpoints, model_list=["Trial"], data=test_data)
    for item in test_data:
        assert "Trial_pred" in item
        del item["Trial_pred"]
        assert "Trial_pred" not in item


def test_add_stat(tmp_path: Path) -> None:
    """Test adding NN models."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for item in np.random.default_rng().random((64, 32)):
        test_data += [{"data": item}]

    # Test built in arguments
    qd = qgain.Detector(stat_tools=[{"name": "StatTrial", "tool": TrialStat}],
                        stats_kwargs=[{"test_param": "Hello."}])
    assert isinstance(qd.controllers["Stat Controller"].get_tool("StatTrial"), TrialStat)
    assert qd.controllers["Stat Controller"].get_tool("StatTrial").msg == "Hello."

    # Test adding manually
    qd = qgain.Detector()
    qd.controllers["Stat Controller"].add_new_tool(stat_tools=[{"name": "StatTrial", "tool": TrialStat}],
                                                   stats_kwargs=[{"test_param": "Hello."}])
    assert isinstance(qd.controllers["Stat Controller"].get_tool("StatTrial"), TrialStat)
    assert qd.controllers["Stat Controller"].get_tool("StatTrial").msg == "Hello."


def test_stat(tmp_path: Path) -> None:
    """Test stat tool usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for item in np.random.default_rng().random((64, 32)):
        test_data += [{"data": item}]

    qd = qgain.Detector(stat_tools=[{"name": "StatTrial", "tool": TrialStat}],
                        stats_kwargs=[{"test_param": "Hello."}])

    qd.define_stat(data=test_data, save=True)
    assert qd.controllers["Stat Controller"].get_tool("StatTrial").params is not None
    assert qd.controllers["Stat Controller"].get_tool("StatTrial").params.shape[0] > 0

    file_paths = list(tmp_path.joinpath("test_exp", "models").rglob("*.pkl"))
    model_checkpoints = []
    for file in file_paths:
        model_checkpoints += [file.name]

    assert len(model_checkpoints) > 0

    # Test not listing any models
    qd.use_models(model_paths=model_checkpoints, data=test_data)
    for item in test_data:
        assert "StatTrial_pred" in item
        del item["StatTrial_pred"]
        assert "StatTrial_pred" not in item

    # Test listing model
    qd.use_models(model_paths=model_checkpoints, model_list=["StatTrial"], data=test_data)
    for item in test_data:
        assert "StatTrial_pred" in item
        del item["StatTrial_pred"]
        assert "StatTrial_pred" not in item


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_metric(tmp_path: Path) -> None:
    """Test plot tool usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for item in np.random.default_rng().random((64, 1)):
        test_data += [{"data": item, "OD_pred": item, "CL_pred": 1, "label": 1}]

    qd = qgain.Detector(od_model=TrialNN, od_dataset_fn=TrialDataset, od_loss_fn=L1Loss, od_aug=True,
                        od_kwargs={"test": "hello"}, cl_model=TrialNN, cl_dataset_fn=TrialDataset, cl_loss_fn=L1Loss,
                        cl_aug=True, cl_kwargs={"test": "hello"})
    qd.data = test_data

    # Check CL
    assert isinstance(qd.controllers["Plot Controller"].get_tool("CL"), PlotterTool)
    assert isinstance(qd.controllers["Plot Controller"].get_tool("OD"), PlotterTool)
    idx = qd.controllers["Plot Controller"].get_id("CL")
    assert qd.controllers["Plot Controller"].saves[idx] is False
    qd.controllers["Plot Controller"].set_save(tool_name="CL", val=True)
    assert qd.controllers["Plot Controller"].saves[idx] is True
    assert qd.controllers["Plot Controller"].styles[idx] is None
    qd.controllers["Plot Controller"].set_style(tool_name="CL", val="default")
    assert qd.controllers["Plot Controller"].styles[idx] == "default"

    # Check OD
    idx = qd.controllers["Plot Controller"].get_id("OD")
    assert qd.controllers["Plot Controller"].saves[idx] is False
    qd.controllers["Plot Controller"].set_save(tool_name="OD", val=True)
    assert qd.controllers["Plot Controller"].saves[idx] is True
    assert qd.controllers["Plot Controller"].styles[idx] is None
    qd.controllers["Plot Controller"].set_style(tool_name="OD", val="default")
    assert qd.controllers["Plot Controller"].styles[idx] == "default"

    qd = qgain.Detector(od_model=TrialNN, od_dataset_fn=TrialDataset, od_loss_fn=L1Loss, od_aug=True,
                        od_kwargs={"test": "hello"}, cl_model=TrialNN, cl_dataset_fn=TrialDataset, cl_loss_fn=L1Loss,
                        cl_aug=True, cl_kwargs={"test": "hello"})
    qd.data = test_data
    qd.plot_metrics(save=True, style="default", plot_kwargs={"CL": {"ground_keys": ["label"]},
                                                             "OD": {"ground_keys": ["data"]}})

    file_paths = []
    for file in list(tmp_path.joinpath("test_exp").rglob("*.png")):
        file_paths += [str(file.name)]
    assert "CL_0.png" in file_paths
    assert "OD_0.png" in file_paths


def test_export(tmp_path: Path) -> None:
    """Test export usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for i in range(10):
        test_data += [{"data": 4, "OD_pred": 5, "CL_pred": 1, "tag": "test", "path": str(tmp_path) + f"_{i}"}]

    qd = qgain.Detector()
    qd.data = test_data

    # test CSV
    qd.export(export_type="csv", keys=["data"])
    files = list(tmp_path.joinpath("test_exp").rglob("*.csv"))

    for file in files:
        with file.open("r") as f:
            data = DictReader(f)

            for idx, sample in enumerate(data):
                assert sample["File"] == str(tmp_path) + f"_{idx}"
                assert sample["Data Tag"] == "test"
                assert sample["CL_pred"] == "1"
                assert sample["OD_pred"] == "5"
                assert sample["data"] == "4"

    # test pickle
    qd.export(export_type="pkl", keys=["data"])
    files = list(tmp_path.joinpath("test_exp").rglob("*.pkl"))
    for file in files:
        with file.open("rb") as f:
            data = pickle.load(f)

            for idx, sample in enumerate(data):
                assert data[sample]["File"] == str(tmp_path) + f"_{idx}"
                assert data[sample]["Data Tag"] == "test"
                assert data[sample]["CL_pred"] == 1
                assert data[sample]["OD_pred"] == 5
                assert data[sample]["data"] == 4

    # test numpy
    qd.export(export_type="numpy", keys=["data"])
    files = list(tmp_path.joinpath("test_exp").rglob("*.npy"))
    for file in files:
        with file.open("rb") as f:
            data = np.load(f, allow_pickle=True).item()

            for idx, sample in enumerate(data):
                assert data[sample]["File"] == str(tmp_path) + f"_{idx}"
                assert data[sample]["Data Tag"] == "test"
                assert data[sample]["CL_pred"] == 1
                assert data[sample]["OD_pred"] == 5
                assert data[sample]["data"] == 4

    # test html
    qd.export(export_type="html", keys=["data"])
    files = list(tmp_path.joinpath("test_exp").rglob("*.html"))
    for file in files:
        with file.open("rb") as f:
            data = read_html(f)

            for idx in range(10):
                assert data[0]["File"][idx] == str(tmp_path) + f"_{idx}"
                assert data[0]["Data Tag"][idx] == "test"
                assert data[0]["CL_pred"][idx] == 1
                assert data[0]["OD_pred"][idx] == 5
                assert data[0]["data"][idx] == 4

    # test hdf
    qd.export(export_type="hdf", keys=["data"])
    files = list(tmp_path.joinpath("test_exp").rglob("*.h5"))
    for file in files:
        with h5py.File(file, "r") as h5_file:

            for idx, sample in enumerate(h5_file):
                assert h5_file[sample].attrs["File"] == str(tmp_path) + f"_{idx}"
                assert h5_file[sample].attrs["Data Tag"] == "test"
                assert h5_file[sample].attrs["CL_pred"] == 1
                assert h5_file[sample].attrs["OD_pred"] == 5
                assert h5_file[sample].attrs["data"] == 4


def test_update(tmp_path: Path) -> None:
    """Test update internal data usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for i in range(10):
        test_data += [{"data": 4, "OD_pred": 5, "CL_pred": 1, "tag": "test", "path": str(tmp_path) + f"_{i}"}]

    qd = qgain.Detector()
    qd.data = test_data

    new_data = []
    for i in range(10):
        new_data += [{"data": 1, "OD_pred": 1, "CL_pred": 2, "tag": "test 2", "path": str(tmp_path) + f"_{i}"}]

    qd.update_samples(new_data)

    for sample in qd.data:
        assert sample["data"] == 1
        assert sample["OD_pred"] == 1
        assert sample["CL_pred"] == 2
        assert sample["tag"] == "test 2"


def test_generate(tmp_path: Path) -> None:
    """Test update internal data usage."""
    qgain.change_path(str(tmp_path))
    qgain.change_exp("test_exp")
    test_data = []
    for i in range(10):
        test_data += [{"data": 4, "OD_pred": 5, "CL_pred": 1, "tag": "test",
                       "path": str(Path("test").joinpath(f"test_{i}.npy"))}]

    qd = qgain.Detector()
    qd.data = test_data

    qd.generate_samples(keys=["OD_pred", "CL_pred", "path"])
    for folder in list(tmp_path.joinpath("test_exp").rglob("DS_*")):
        assert folder.joinpath("data", "data_info", "data_roster.h5").is_file()
        with h5py.File(folder.joinpath("data", "data_info", "data_roster.h5"), "r") as h5_file:
            for i, ds in enumerate(h5_file):
                assert h5_file[ds].attrs["path"] == f"test/test_exp_{i}.h5"
                assert h5_file[ds].attrs["tag"] == "test"
        for i in range(10):
            assert folder.joinpath("data", "data_files", "test", f"test_exp_{i}.h5").is_file()
            with h5py.File(folder.joinpath("data", "data_files", "test", f"test_exp_{i}.h5"), "r") as h5_file:
                assert h5_file.attrs["path"] == f"test/test_{i}.npy"
                assert h5_file.attrs["tag"] == "test"
                assert h5_file.attrs["CL_pred"] == 1
                assert h5_file.attrs["OD_pred"] == 5
                assert h5_file["data"][()] == 4
