# Q-GAIN Modularity

Aspects of Q-GAIN can be replaced or augmented by replacing the function calls in the initialization process of any class that inherits the Q-GAIN base Detector class. These will override the default behavior of the detector to enable usage on data outside the scope of the built in modules. 

Currently you can modify the following:

* The data preprocessing function when importing new data, process_fn.
* The ML object detector, od_model.
* The ML classifier, cl_model.
* Additional ML models via the ML controller's add_new_tool() method.
* The pytorch dataset handler for the object detector data, od_dataset_fn.
* The pytorch dataset handler for the classifier data, cl_dataset_fn.
* The loss function used when training the object detector, od_loss_fn.
* The loss function used when training the classifier, cl_loss_fn.
* The statistical based models to use for analysis, stat_tools.
* Additional statistical based models via the metric controller's add_new_tool() method.

ML models and stat tools can also be provided optional kwarg arguments via their corresponding kwarg dictionary arguments.
Further customization can be achieved by overriding the parent methods of a detector. New functionality can also be added by creating new controller types.

Examples of building new detectors and using them can be found in the Examples repository located on the [Q-GAIN GitHub organization](https://github.com/Q-GAIN/Examples).

## Processing Data for Import
Before data can be loaded into a detector and analyzed it must be imported. This prepares the data and attaches any relevant metadata information. This process need only be done once and data can then be repeatedly loaded into any future detectors that point to that experiment directory.

The processing function should be written to import data such that it satisfies Q-GAIN's requirements to function, or the requirements of any modules you replace. Replacing the processing functionality with your own can be done by providing a callable function to the *process_fn* argument during initialization.

This function should at minimum contain the following keys for each data point:

- 'tag' : A descriptor of the data. This is used by Q-GAIN to determine which data to load.
- 'data' : Target measurement data
- 'sub_dir' : The directory a sample should reside in.

Any additional metadata keys in the dictionary will be saved as additional attributes.

The keys 'tag' and 'path' (derived from 'sub_dir') will be saved to the roster file. These will also be saved as attributes to the data point's HDF5 file. The 'data' entry will be saved as a separate dataset in the sample's HDF5 file. Any other keys will be saved as attributes to the HDF5 file. If these happen to be a dictionary these will be saved as an empty dataset whose attributes are the dictionary items.

The key 'sub_dir' determines what sub directories are created in the data path.

If the key 'filename' is found then HDF data point files are saved with that naming scheme. Otherwise the files will be saved prepended with the current experiment name.

```python
from qgain import Detector
import numpy as np

def dummy_proc_fn(data_path: str, labels: str, *, augment: bool):
    data = np.load(data_path)
    data_samples = []
    for idx, item in enumerate(data):
        data_sample = {}
        data_sample['tag'] = labels
        data_sample['sub_dir'] = "DummyData"
        data_sample['data'] = item
        data_sample['filename'] = f"{data_path}_{idx}"
        data_samples += [data_sample]   
    return data_samples
    
class DummyDetector(Detector):
    def __init__:
        super().__init__(process_fn=dummy_proc_fn)
```

## Adding ML Tools
The Q-GAIN library has a controller class that is tasked with running training and inference on PyTorch compatible models. Specifying models for the built in object detection and classification tools can be done by providing a callable pytorch model during initialization to the arguments *od_model* and *cl_model*. These models should have the typical pytorch forward function that accepts a tensor input. The ML controller class will invoke these models, pass any needed arguments provided by you, and call them for training and inference. 

The output of these models should be in a form the provided loss function expects, including the target tensors you're training against. The loss functions used by the controllers can be changed by providing a callable pytorch module appropriate for generating a loss value to the arguments *od_loss_fn* and *cl_loss_fn*.

Finally, the data expected by these models requires a custom dataset function. You can specify one with the *od_dataset_fn* and *cl_dataset_fn* arguments. At minimum these should contain __len__ and __getitem__ function definitions for proper functionality with a pytorch DataLoader. The ML controller of Q-GAIN expects the output of the datasets to be in the form of (data, target). Note that if you do not have an argument called 'augment' you should specify od_aug = None, cl_aug = None, or augment = None when initializing a Detector object.
```python
from qgain import Detector
import torch

class DummyModel(torch.nn.Module):
    def __init__():
        super().__init__()
        self.layer = torch.nn.Linear(128, 64)
    def forward(self, x):
        return self.layer(x)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data: list):
        self.data = []
        self.label = []
        for item in data:
            self.data += [torch.from_numpy(item['data'])]
            self.label += [torch.from_numpy(item['label'])]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data, self.label

class DummyDetector(Detector):
    def __init__:
        super().__init__(od_model=DummyModel, od_dataset_fn=DummyDataset, 
                         od_loss_fn=torch.nn.SmoothL1Loss)
```

The ML controller can add additional ML tools by using its add_new_tool() method. This allows one to add aditional ML analysis types. This method expects similar arguments to be passed to it. It also expects a name to be specified for the tool which the controller references for other parts of the detector.
```python
class DummyDetector(Detector):
    def __init__:
        super().__init__()
        self.controllers["ML Controller"].add_new_tool(model=DummyModel, name="DummyModel", dataset_fn=DummyDataset, loss_fn=torch.nn.SmoothL1Loss)
```
## Adding Statistical Based Tools
The Q-GAIN library also has a controller class that maintains all functionality for the statistical based analysis tools. These can be added by passing dictionaries to the detector initialization arguments *stat_tools* and *stats_kwargs*. A callable function should be provided to the key "tool" and a name given to the "name" key for the *stat_tools* dictionary. Any required function argument parameters should be passed as a keyword dictionary to *stats_kwargs*

New analysis tools should either be a class which wraps an analysis algorithm or a class which contains fit and transform methods. The controller expects that the transform method return its results in a list like object which Q-GAIN will use to assign results to its data entries.

```python
from qgain import Detector
import numpy as np
from sklearn.cluster import KMeans

class DummyMetric:
    def __init__(self, clusters: int):
        self.clusters = clusters
        self.kmeans = KMeans(n_clusters=self.clusters)
    def fit(self, data: list[dict]) -> None:
        x = []
        for item in data:
            x += [item['data'].flatten()]
        self.kmeans.fit(np.array(x))
    def transform(self, data: list[dict]) -> list:
        res = []
        for item in data:
            res += [self.kmeans.transform(item['data'].flatten().reshape(1, -1))]   
        return res

class DummyDetector(Detector):
    def __init__:
        super().__init__(stat_tools=[{"name": "KMeans", "tool": DummyMetric}],
                         stats_kwargs=[{"clusters":5}])
```

These analysis methods can also be added by using the controller's add_new_tool() method in a similar fashion to the ML models. When invoking this method it uses the same arguments and constraints as outlined previously for *stat_tools* and *stats_kwargs*.