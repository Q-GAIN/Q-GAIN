# Q-GAIN
Q-GAIN is a Python library serving as the foundation of a growing modular architecture to support a wider range of machine learning and physically informed analysis applications. 

It is featured with the original SolDet package as a submodule which includes classification, object detection, and physics informed metric methods. Technical details of the original package are explained in https://arxiv.org/abs/2111.04881. The dataset used to prepare the original SolDet and used by the modern SolDet module is available at https://data.nist.gov/od/id/mds2-2363. It also includes a vortex detector as a submodule for identifying the positions of vortices in Bose-Einstein Condensates.

## Installing Q-GAIN

Installing Q-GAIN requires python to be >= 3.8.

To install qgain it is recommended to create a fresh environment.
* For regular python you can make use of venv to create a virtual environment. Create a folder for the environment and pass the path to this directory to venv.
    ```console
    python -m venv /Users/User1/qgainEnv
    ```
    You must then activate it, which is dependent on platform. For systems with bash this can be done with the activate script.
    ```bash
    source path_to_env/bin/activate
    ```
    For systems with a Windows console you can use the activate batch script.
    ```console
    path_to_env/bin/activate.bat
    ```

* For Anaconda you can use its built in environment manager.
    ```console
    conda create --name qgainEnv python=3.11
    ```
    You must then activate it, which can be done with the activate command.
    ```console
    conda activate qgainEnv
    ```

Q-GAIN can currently be installed via GiHub.

First, clone or download the github repository and place it in your desired directory.  Assuming the usage of git you can use the git command in Anaconda, Python, or your system environment.
```console
git clone url_to_repository
cd path_to_qgain_dir
```
Change to this directory before installing Q-GAIN. From the console invoke pip and install in editable mode with the option -e.
```console
pip install -e qgain
```

## Getting started

The main features of Q-GAIN are accessed with the `Detector` class, and some helper functions are available as part of the Q-GAIN library to assist with its usage.

To start, import Q-GAIN and create a detector object. Q-GAIN comes with two types by default: one for solitons and one for vortices. For solitons the SolDet sub package can be used. This will create a SolitonDetector object, which inherits the functionality of Q-GAIN's Detector class. Similarly, for vortices the vdet subpackage can be used to create a VortexDetector object.

```python
from qgain.soldet import soliton_detector
from qgain.vdet import vortex_detector

sd = soliton_detector.SolitonDetector()
vd = vortex_detector.VortexDetector()
```

On import, or when creating a detector object, Q-GAIN will check to confirm if a configuration file, CONFIG.ini, exists in its package path. If not this is created with default values. 

[PATHS]\
data_path = \<path to home directory\>/qgain\
def_exp_name = default_ds

These set up the target directories for the required folder structure to run Q-GAIN. The <em>data_path</em> points to the directory all experimental data folders will reside in. An experiment can be specified with <em>def_exp_name</em>, which will set the target directory for where Q-GAIN's class data will be saved to. The default structure is,

data_path\
|- def_exp_name\
    |- data\
        |- data_files\
        |- data_info\
    |- models\
|- ...

Class data folders reside within the <em>data_files</em> directory, and their naming depends on the information contained within the roster file, residing within the <em>data_info</em> directory. Q-GAIN treats classes as a value which seperates one data from another.

Multiple experiment folders can reside in the data path, and any detector objects created will reference the current def_exp_name. Changing to a different experiment folder requires creating another detector object.

Although you can modify the configuration file manually, helper functions exist to do this for you.

To change the data path you can use the utility function change_path().


```python
qgain.change_path('../Documents/Machine Learning/Detectors/')
```

To change the current experiment directory you can use change_exp().


```python
qgain.change_exp('TestExp')
```

Although by default Q-GAIN will create the necessary folder structure if it does not exist, you may also manually trigger this creation with the use of the config() function. This will also create the CONFIG file if it is not found.


```python
EXP_PATH, EXP_NAME = qgain.config()
print(EXP_PATH)
print(EXP_NAME)
```

## Loading and Importing Data

To import new data into the class directories you can call import_data() on any detector object. This requires a path to be given where the new data is located. By default this uses the Q-GAIN preprocessing function, which can be modified to support other HDF hierarchy    configurations by modifying the appropriate argument.

For more information on pre-processing functions see the full documentation on process_data().

```python
sd = qgain.soldet.soliton_detector.SolitonDetector()
sd.import_data(path='../BEC_data_2023_0613/0001', target='xy', atoms_name='atoms', bg_name='background', probe_name='probe', label=9, width=164, height=132)
```

To load in preprocessed, imported data into any detector object you can use load_data(). This will load all files located in the roster file. You can specify which classes to load with the *labels* argument. This function will also scale the data to be between 0 and 1 and can be controlled with the *minmax* argument. The amount of data to use for training the ML models can be specified with the argument *data_frac*.


```python
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
```


You may optionally use the helper function download_ds() from the SolDet module to download the original SolDet dataset for use with any SolitonDetector. Note that this will also create the necessary folder structure if it does not exist in your experiment path.


```python
from qgain.soldet import soliton_datasets

soliton_datasets.download_ds()
```

The original SolDet dataset format is no longer supported in the newest Q-GAIN library. The library now expects HDF5 files of the following structure, at minimum:

* A HDF Group named after the filename
    * Group attributes containing information about the data.
        * label - The class label.
        * original_file - The filename of the old Q-GAIN file.
        * Other relevant attributes. 
    * data - Dataset holding the target data.

For more information on the expected structure of the HDF files, please see full documentation on import_data, process_data, and soldet_to_h5.

If you are still using the old SolDet format, a helper function is available to convert the dataset to the new format, soldet_to_h5().


```python
from qgain.soldet import soliton_datasets

soliton_datasets.soldet_to_h5(exp_path, delete_old = True)
```

## Training The Models

To train the ML models you can use the Detector function train_nn(). The argument *model_list* specifies which models to train on the data. This expects a list of the available models. The arguments *epochs* and *patience* control how long to train the models. You can also set the optimizer used and its corresponding learning rate. The trained weights will be saved to the models folder in your experiment directory. 

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.train_nn(model_list=['classifier', 'object detector'], patience=10, epochs=50)
```

The default state of Q-GAIN is to use the data loaded into the current detector object. However, if you require a different set of data to train from you can change the argument *data* from None to the desired set of data. Note that this list or dictionary of samples must have the structure anticipated by Q-GAIN.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
tr_set = []
for item in sd.data:
    if item["label"] == 1:
        tr_set.append({"data":item["data"], "label":item["label"], "positions":item["positions"]})

sd.train_models(model_list=["classifier", "object detector"], patience=10, epochs=50, data=tr_set)

```

## Building Statistical Metrics

Q-GAIN comes with the ability to deploy statistical based analysis methods. These models typically require building a metric to transform the data before doing any relevant calculations. The fitting portion of these models can be invoked with the Q-GAIN detector method *build_pi*, which will work through every metric loaded into the controller and call their corresponding fit functions.
Optionally, if the argument save is set to True these metrics can be saved as a file in the models folder of the experiment path.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.define_pi(save=True)
```

## Making Use of The Models

To use any of these models you can invoke the use_models() function. Specifying any of the options 'classifier', 'object detector', 'pie classifier', or 'quality estimator' in the argument *model_list* will make the function use those features. The argument *model_paths* can be used to dictate the trained weights or metric files which are located in the models folder of the experiment path. Results are saved in the data dictionary for each sample.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[1], data_frac=0.9, minmax=[-1, 3])

sd.use_models(model_list=['classifier', 'object detector', 'pie classifier', 'quality estimator'], 
              model_paths=['20240318_215015_CL.pt', '20240318_225331_OD.pt', 
                            '20241210_223439_PIE_classifier.pkl', '20241210_223508_QE.pkl'])      
```

## Q-GAIN Modularity

Aspects of Q-GAIN can be replaced or augmented by replacing the function calls in the initialization process of any class that inherits the Q-gain base Detector class. These will override the default behavior of the library to enable usage on data outside the scope of the built in modules. 

Currently you can modify the following:

* The data preprocessing function when importing new data, process_fn.
* The ML object detector, od_model.
* The ML classifier, cl_model.
* Additional ML models via the ML controller's add_new_tool() method.
* The pytorch dataset handler for the object detector data, od_dataset_fn.
* The pytorch dataset handler for the classifier data, cl_dataset_fn.
* The loss function used when training the object detector, od_loss_fn.
* The loss function used when training the classifier, cl_loss_fn.
* The statistical based models to use for analysis, pi_metrics.
* Additional statistical based models via the metric controller's add_new_metric() method.

### Processing Data for Import
The processing function should be written to import data such that it satisfies Q-GAIN's requirements to function, or the requirements of any modules you replace. Replacing the processing functionality with your own can be done by providing a callable function to the *process_fn* argument during initialization.
```python
from qgain import Detector
import numpy as np

def dummy_proc_fn(data_path: str, labels: str, *, augment: bool):
    data = np.load(data_path)
    data_samples = []
    for item in data:
        data_sample = {}
        data_sample['label'] = labels
        data_sample['filename'] = data_path
        data_sample['class_dir'] = "Experimental"
        data_sample['data'] = item
        data_samples += [data_sample]   
    return data_samples
    
class DummyDetector(Detector):
    def __init__:
        super().__init__(process_fn=dummy_proc_fn)
```

### Replacing ML Models
The Q-GAIN library has a controller class that is tasked with running training and inference on PyTorch compatible models. Replacing the default ML tools in Q-GAIN can be done by providing a callable pytorch model during initialization. These models should have the typical pytorch forward function that accepts a tensor input. The ML controller class will invoke these models, pass any needed arguments provided by you, and call them for training and inference. 

The output of these models should be in a form the provided loss function expects, including the target tensors you're training against. The loss functions used by the controllers can be changed by providing a callable pytorch module appropriate for generating a loss value.

Finally, ff the data expected by these models requires a custom dataset function, you can specify one with the *od_dataset_fn* and *cl_dataset_fn* arguments. At minimum these should contain __len__ and __getitem__ function definitions for proper functionality with a pytorch DataLoader. The ML controller of Q-GAIN expects the output of the datasets to be in the form of (data, target). Note that if you do not have an argument called 'augment' you should specify od_aug = None, cl_aug = None, or augment = None when initializing a Detector object.
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
        self.ml_top.add_new_tool(model=DummyModel, name="DummyModel", 
                                 dataset_fn=DummyDataset, loss_fn=torch.nn.SmoothL1Loss)
```