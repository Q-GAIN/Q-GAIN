# Q-GAIN
Q-GAIN is a Python library serving as the foundation of a growing modular architecture to support a wider range of machine learning and physically informed analysis applications.

It is featured with the original SolDet package as a submodule which includes classification, object detection, and physics informed metric methods. Technical details of the original package are explained in https://arxiv.org/abs/2111.04881. The dataset used to prepare the original SolDet and used by the modern SolDet module is available at https://data.nist.gov/od/id/mds2-2363. It also includes a vortex detector as a submodule for identifying the positions of vortices in Bose-Einstein Condensates.

## Installing Q-GAIN

Installing Q-GAIN requires python to be between 3.8 and 3.12.

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

There are two ways you can install Q-GAIN into your environment, depending on your needs.

* Install Q-GAIN directly via pip. 
    ```console
    pip install qgain
    ```

* Install the Q-GAIN locally via the GitHub repository.

    This option is more suitable for when there is a need to modify or contribute back to the Q-GAIN library.

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

The main features of Q-GAIN are accessed with the `Detector` class, and some helper functions are available as part of the Q-GAIN library to assist with its usage. The following will be a demonstration of using the SolDet subpackage of Q-GAIN for soliton detection.

To start, import soldet and create a SolitonDetector object, which inherits the functionality of Q-GAIN's Detector class.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
```

On import, or when creating the SolitonDetector object, Q-GAIN will check to confirm if a configuration file, CONFIG.ini, exists in its package path. If not this is created with default values. 

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

Class folders reside within the <em>data_files</em> directory, and their naming depends on the information contained within the roster file, residing within the <em>data_info</em> directory. The default SolDet class structure is,

data_files\
    |- class-0\
    |- class-1\
    |- class-2\
    |- class-8\
    |- class-9

Multiple experiment folders can reside in the data path, and any detector objects created will reference the current def_exp_name. Changing to a different experiment folder requires creating another detector object.

Although you can modify the configuration file manually, helper functions exist to do this for you.

To change the data path you can use the utility function change_path().


```python
qgain.change_path('../Documents/Machine Learning/SolitonDetector/')
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

    User path not found, creating.
    ../Documents/Machine Learning/SolitonDetector/TestExp
    TestExp

## Loading and Importing Data

You may optionally use the helper function download_ds() from the SolDet module to download the original SolDet dataset. Note that this will also create the necessary folder structure if it does not exist in your experiment path.


```python
from qgain.soldet import soliton_datasets

soliton_datasets.download_ds()
```

    Downloading SolDet data. This may take a while. Please wait..


    Downloading data_info.zip.: 100%|██████████| 1.48M/1.48M [00:00<00:00, 23.3MB/s]


    Extracting data. Please wait..


    Downloading data_files.zip.: 100%|██████████| 4.03G/4.03G [01:28<00:00, 49.1MB/s]


    Extracting data. Please wait..


The original SolDet dataset format is no longer supported in the newest Q-GAIN library. The library now expects HDF5 files of the following structure, at minimum:

* A HDF Group named after the filename
    * Group attributes containing information about the data.
        * label - The class label.
        * original_file - The filename of the old Q-GAIN file.
        * Other relevant attributes. 
    * data - Dataset holding the target data.

For more information on the expected structure of the h5 files, please see full documentation on import_data, process_data, and soldet_to_h5.

If you are still using the old SolDet format, a helper function is available to convert the dataset to the new format, soldet_to_h5().


```python
from qgain.soldet import soliton_datasets

soliton_datasets.soldet_to_h5(exp_path, delete_old = True)
```

    Converting data..: 100%|██████████| 16478/16478 [25:38<00:00, 10.71it/s]


To load in preprocessed data into the SolitonDetector you can use load_data() from the Q-GAIN library. This will load all files located in the roster file and you can specify which classes to load with the *labels* argument. This function will also scale the data to be between 0 and 1 and can be controlled with the *minmax* argument. For the SolDet dataset this should be set to (-1, 3). The amount of data to use for training the ML models can be specified with the argument *data_frac*.


```python
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
```

    Loading processed data..: 100%|██████████| 16478/16478 [00:59<00:00, 275.71it/s]
    Normalizing Data..: 100%|██████████| 4342/4342 [00:00<00:00, 18655.75it/s]

To import new data into the class folders you can call import_data(). This requires a path to be given where the new data is located. By default this uses the Q-GAIN preprocessing function with SolDet parameters, but this can be modified to support other Labscript folder configurations by modifying the appropriate argument.

For more information on pre-processing functions see the full documentation on process_data().

```python
sd.import_data(path='../BEC_data_2023_0613/0001', target='xy', atoms_name='atoms', bg_name='background', probe_name='probe', label=9, width=164, height=132)
sd.load_data(labels=[9], data_frac=0.9, minmax=[-1, 3])
```

    Getting Raw Data..: 100%|██████████| 50/50 [00:01<00:00, 31.44it/s]
    Processing Raw Data..: 100%|██████████| 50/50 [00:49<00:00,  1.01it/s]
    Writing data files..: 100%|██████████| 50/50 [00:00<00:00, 77.12it/s]
    Loading processed data..: 100%|██████████| 50/50 [00:00<00:00, 111.12it/s]
    Normalizing Data..: 100%|██████████| 50/50 [00:00<00:00, 12369.66it/s]

## Training The Models

To train the ML models you can use the Detector function train_nn(). The argument *model_list* specifies which models to train on the data. This expects a list of the available models, 'classifier' or 'object detector'. The arguments *epochs* and *patience* control how long to train the models. The trained weights will be saved to the models folder in your experiment directory. 

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.train_nn(model_list=['classifier', 'object detector'], patience=10, epochs=50)
```

    Loading processed data..: 100%|██████████| 16478/16478 [00:31<00:00, 520.28it/s] 
    Normalizing Data..: 100%|██████████| 4342/4342 [00:00<00:00, 14533.09it/s]

    Dataset loaded for Classifier.
    Classifier model run: 1

    Epoch: 50/50 | Loss: 0.000034 | Test Loss: 0.002480 | Acc.: 0.999107:  98%|█████████▊| 49/50 [08:36<00:10, 10.55s/it]

    Done! Minimum Test Loss: 0.0010950952629952683 with Accuracy: 1.0.
    Dataset loaded for Object Detector.
    
    Object Detector model run: 1

    Epoch: 50/50 | Loss: 0.672539 | Avg. Test Loss: 1.913660 | Acc.: 0.955682: 100%|██████████| 50/50 [09:18<00:00, 11.17s/it] 

    Done! Minimum Test Loss: 1.8960855658608253 with Accuracy: 0.9388888888888889.

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

## Building The Physics Informed Models

To implement the conventional analysis methods used by SolDet, the subpackage takes advantage of Q-GAIN's metric controller to deploy its algorithms, only requiring SolDet to list its analysis modules during initialization.
```python
class SolitonDetector(Detector):
    def __init__(self, od_kwargs: dict | None = None, cl_kwargs: dict | None = None, 
                 *, augment: bool = True) -> None:
                 
        od_kwargs = {} if od_kwargs is None else od_kwargs
        cl_kwargs = {"num_classes": 3} if cl_kwargs is None else cl_kwargs
        
        super().__init__(process_fn=process_data, od_model=ObjectDetector, 
                         od_dataset_fn=SolitonODDataset, od_loss_fn=MetzLoss, 
                         cl_model=MLST2021CNNmodern, cl_dataset_fn=SolitonClassDataset,
                         cl_loss_fn=NLLLoss, augment=augment, 
                         od_kwargs=od_kwargs, cl_kwargs=cl_kwargs,
                         pi_metrics=[{"name": "pie classifier", "metric": PIEClassifier},
                                       {"name": "quality estimator", "metric": QE}],
                         pi_kwargs=[{"func": "modern", "transformer": PowerTransformer},
                                      {"func": "modern", "transformer": PowerTransformer}])
```
By passing in a callable function and a name to address it by to the controller via the *pi_metrics* argument, and any optional arguments to initialize the modules via the *pi_kwargs* argument, the controller automates its deployment in the Q-GAIN pipeline.

These models require building a metric to transform the data before doing any of the relevant calculations. The fitting portion of these models can be invoked with the Q-GAIN detector method *build_pi*, which will work through every metric loaded into the controller and call their corresponding fit functions.
Although not required, the SolDet subpackage invokes this method for each of its PIE classifier and quality estimator methods for convenience. For these functions the default values are sufficient, and the final result will be saved to the detector. Optionally, if the argument save is set to True these metrics can be saved as a file in the models folder of the experiment path.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.define_pie_classifier(save=True)
sd.define_quality_estimate(save=True)
```

    Loading processed data..: 100%|██████████| 16478/16478 [00:41<00:00, 398.73it/s] 
    Normalizing Data..: 100%|██████████| 4342/4342 [00:00<00:00, 22647.95it/s]


    Building PIE metric.


    100%|██████████| 3212/3212 [00:42<00:00, 76.28it/s]


    Building QE metric.


    100%|██████████| 2229/2229 [00:28<00:00, 78.82it/s]


## Making Use of The Models

To use any of these models you can invoke the use_models() function of the SolitonDetector class. Specifying any of the options 'classifier', 'object detector', 'pie classifier', or 'quality estimator' in the argument *model_list* will make the function use those features. The argument *model_paths* can be used to dictate the trained weights or metric files which are located in the models folder of the experiment path. Results are saved in the data dictionary for each sample.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(labels=[1], data_frac=0.9, minmax=[-1, 3])

sd.use_models(model_list=['classifier', 'object detector', 'pie classifier', 'quality estimator'], 
              model_paths=['20240318_215015_classifier.pt', '20240318_225331_object.pt', 
                             '20241210_223439_PIE_classifier.pkl', '20241210_223508_QE.pkl'])      
```

    Loading processed data..: 100%|██████████| 16478/16478 [00:18<00:00, 891.10it/s] 
    Normalizing Data..: 100%|██████████| 3212/3212 [00:00<00:00, 22169.88it/s]

    Loaded ../Documents/Machine Learning/SolitonDetector/TestExp/models/20240318_215015_classifier.pt.
    Loaded ../Documents/Machine Learning/SolitonDetector/TestExp/models/20240318_225331_object.pt.
    Loaded ../Documents/Machine Learning/SolitonDetector/TestExp/models/20241210_223439_PIE_classifier.pkl.
    Loaded ../Documents/Machine Learning/SolitonDetector/TestExp/models/20241210_223508_QE.pkl.
    Starting ML Classifier.
    Classifier model loaded.
    Running model, please wait..

    Running..: 100%|██████████| 3212/3212 [00:44<00:00, 71.86it/s]

    Finished.
    Starting ML Object Detector.
    Object Detector model loaded.
    Running model, please wait..

    Running..: 100%|██████████| 3212/3212 [00:18<00:00, 169.40it/s]

    Finished.
    Starting Physics Informed Classifier.

    PIE Classifier running..: 100%|██████████| 3212/3212 [01:55<00:00, 27.81it/s]

    Starting Physics Informed Quality Estimator.
    
    Quality Estimate running..: 100%|██████████| 3212/3212 [00:50<00:00, 64.01it/s]

## Q-GAIN Modularity

Aspects of Q-GAIN can be replaced or augmented by replacing the function calls in the initialization process of any class that inherits the Q-gain base Detector class. These will override the default behavior of the library to enable usage on data outside the scope of the built in modules. 

Currently you can modify the following:

* The data preprocessing function when importing new data, process_fn.
* The ML object detector, od_model.
* The ML classifier, cl_model.
* The pytorch dataset handler for the object detector data, od_dataset_fn.
* The pytorch dataset handler for the classifier data, cl_dataset_fn.
* The loss function used when training the object detector, od_loss_fn.
* The loss function used when training the classifier, cl_loss_fn.
* The statistical based models to use for analysis, pi_metrics.

### Processing Data for Import
The processing function should be written to import data such that it satisfies Q-GAIN's requirements to function, or the requirements of any modules you replace. When using Q-GAIN's processing function to import new data, such as labscript data organized differently, you can change the way the image data is retrieved by passing in new values to the arguments of [import_data()](qgain.detector.Detector.import_data) of Q-GAIN.

```python
sd.import_data(path='../MOT_Exploration_2024/0001', target='images', atoms_name='MOT', bg_name='MOT_DARK', probe_name='MOT_PROBE', label=9, width=164, height=132)
```

Replacing the processing functionality with your own can be done by providing a callable function to the *process_fn* argument during initialization.
```python
from qgain.Detector import Detector

def dummy_proc_fn(dir: str, labels: str, *, augment: bool) -> list[dict]:
    # Do Stuff
    return data_samples

qd = Detector(process_fn=dummy_proc_fn)
qd.import_data(path='../Exp/001')
```

### Replacing ML Models
The ML models used by Q-GAIN can be replaced by providing a callable pytorch model during initialization. These models should have the typical pytorch forward function that accepts a tensor input. The [ObjectControl](qgain.run_od.ObjectControl) and [ClassifierControl](qgain.run_classifier.ClassifierControl) classes will invoke these models, pass any needed arguments provided by you, and call them for training and inference.
```python
from qgain.Detector import Detector
import torch

class DummyDetector(torch.nn.Module):
    def __init__(self, dropout: float, layers: int, compression: list) -> None:
        super().__init__()
        # Do Stuff

    def forward(self, x: Torch.Tensor) -> Torch.Tensor:
        #Do Stuff
        return x

params = {'dropout': 0.1, 'layers': 4, 'compression': [1, 16, 32, 64]}

qd = Detector(od_model=DummyDetector, od_kwargs=params) 
```

The output of these models should be in a form the provided loss function expects, including the target tensors you're training against. The loss functions used by the controllers can be changed by providing a callable pytorch module appropriate for generating a loss value.
```python
from qgain.Detector import Detector
import torch

class DummyDetector(torch.nn.Module):
    def __init__(self, dropout: float, layers: int, compression: list) -> None:
        super().__init__()
        # Do Stuff

    def forward(self, x: Torch.Tensor) -> Torch.Tensor:
        #Do Stuff
        return x

params = {'dropout': 0.1, 'layers': 4, 'compression': [1, 16, 32, 64]}

qd = Detector(od_model=DummyDetector, od_kwargs=params, od_loss_fn=torch.nn.MSELoss) 
```

If the data expected by these models requires a custom dataset function, you can specify one with the *od_dataset_fn* and *cl_dataset_fn* arguments. At minimum these should contain length and indexing magic function definitions for proper functionality with a pytorch DataLoader. The ML controllers of Q-GAIN expect the output of the datasets to be in the form of (data, target). Note that if you do not have an argument called 'augment' you should specify augment = None when initializing a Detector object.

```python
from qgain.Detector import Detector 
import torch

class DummyDetector(torch.nn.Module):
    def __init__(self, dropout: float, layers: int, compression: list):
        super().__init__()
        # Do Stuff

    def forward(self, x):
        #Do Stuff
        return x

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data: list) -> None:
        #Do Stuff
            
    def __len__(self) -> int:
        #Do Stuff
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        #Do Stuff

        return data, label

params = {'dropout': 0.1, 'layers': 4, 'compression': [1, 16, 32, 64]}

qd = Detector(od_model=DummyDetector, od_kwargs=params, od_loss_fn=torch.nn.MSELoss, od_dataset_fn=DummyDataset, augment=None) 
```

### Modularity Demonstration using MNIST

One of Q-GAIN's main design goals is to enable rapid deployment of machine learning based analysis tools. To demonstrate this we will use it to quickly develop a simple tool to act on a standard example such as the MNIST handwritten digits dataset.

First lets get the data directories for this new experiment example set up.

```python
import qgain

qgain.change_exp('MNIST')
qgain.change_path('/Users/qgain')
exp_path, exp = qgain.config()
```

    User path not found, creating.

We will be creating a new detector class we'll call MNISTDetector. We will inherit the functionality of Q-GAIN's Detector class and add functionality of our own. We will do this by passing in custom calls to the initialization process. 
We will make use of the classifier model from the SolDet subpackage, but modify it to use 10 class labels instead of 3.

```python
import qgain
from qgain.Detector import Detector

class MNISTDetector(Detector):
    def __init__(self) -> None:
        super().__init__(cl_model=qgain.soldet.classifier_nn.MLST2021CNNmodern,
                        cl_kwargs={"num_classes": 10})

```

Since the SolDet classifier will be adapted to run with MNIST we will need to create a new processing function to prepare the data for importing into new class folders. These class folders will be the digits themselves, 0 - 9, and Q-GAIN will be informed of this by making use of the 'class_dir' key during import. The digit labels themselves will be passed to the 'label' key. Q-GAIN also requires a filename entry, which we will just set to 'None'.

We will further identify the data with a label of training or validation, signified by the train boolean flag. This will be saved to the data dictionary as additional meta information.

```python
import numpy as np
import torch
from qgain.detector import Detector
from torchvision.datasets import MNIST
from tqdm import tqdm

class MNISTDetector(Detector):
    def __init__(self) -> None:
        super().__init__(process_fn=self.mnist_process_fn,
                        cl_model=qgain.soldet.classifier_nn.MLST2021CNNmodern,
                        cl_kwargs={"num_classes": 10})

    def mnist_process_fn(self, input_data: MNIST, *, train: bool = True) -> list[dict]:
        data_samples = []
        data_set = MNIST(input_data, train=train, download=True)
        for (image, label) in tqdm(data_set, desc="Processing data"):
            sample = {}
            sample["data"] = np.array(image)
            sample["label"] = label
            sample["class_dir"] = f"class-{label}"
            sample["training"] = train
            sample["filename"] = "None"
            data_samples += [sample]
        return data_samples

mnsit_det = MNISTDetector()
mnsit_det.import_data(exp_path, **{'train': True})
mnsit_det.import_data(exp_path, **{'train': False})
```

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to /Users/qgain/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
    100%|██████████| 9912422/9912422 [00:00<00:00, 41760459.52it/s]
    Extracting /Users/qgain/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to /Users/qgain/MNIST/MNIST/raw

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to /Users/qgain/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
    100%|██████████| 28881/28881 [00:00<00:00, 3654389.22it/s]
    Extracting /Users/qgain/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to /Users/qgain/MNIST/MNIST/raw

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to /Users/qgain/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
    100%|██████████| 1648877/1648877 [00:00<00:00, 27373624.16it/s]
    Extracting /Users/qgain/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to /Users/qgain/MNIST/MNIST/raw

    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to /Users/qgain/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
    100%|██████████| 4542/4542 [00:00<00:00, 1898786.88it/s]
    Extracting /Users/qgain/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to /Users/qgain/MNIST/MNIST/raw

    Processing data..: 100%|██████████| 60000/60000 [00:02<00:00, 25486.82it/s]
    Writing data files..: 100%|██████████| 60000/60000 [52:22<00:00, 19.09it/s]  
    Processing data..: 100%|██████████| 10000/10000 [00:00<00:00, 23534.11it/s]
    Writing data files..: 100%|██████████| 10000/10000 [11:01<00:00, 15.12it/s]

Now we can load the data into the detector using [load_data()](qgain.detector.Detector.load_data). We want all the digits so we list off 0 - 9 in the class labels. The data also needs to be rescaled to be between 0 and 1 so *scale* is set to True and the pixel value range is given to the argument *minmax*. Lastly, we want a 90/10 split for the data we'll train on so *data_frac* is set to 0.9.

```python
mnsit_det.load_data(labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], data_frac=0.9, minmax=[0, 255], scale=True)
```

    Loading processed data..: 100%|██████████| 70000/70000 [03:19<00:00, 351.13it/s]
    Normalizing Data..: 100%|██████████| 70000/70000 [00:00<00:00, 103637.38it/s]

Although the built in SolDet classifier dataset would work if the augmentation was turned off, we will override it with our own for demonstration purposes. This will then be passed in as an argument to the initialization of the new detector as was done before. Since there's no need for augmentation the argument *augment* is set to None so Q-GAIN knows not to look for this.

```python
import numpy as np
import torch
from qgain.detector import Detector
from torchvision.datasets import MNIST
from tqdm import tqdm

class MNISTDetector(Detector):
    def __init__(self) -> None:
        super().__init__(process_fn=self.mnist_process_fn,
                        cl_model=qgain.soldet.classifier_nn.MLST2021CNNmodern,
                        cl_kwargs={"num_classes": 10},
                        cl_dataset_fn=self.MNISTDataset,
                        augment=None)

    def mnist_process_fn(self, input_data: MNIST, *, train: bool = True) -> list[dict]:
        data_samples = []
        data_set = MNIST(input_data, train=train, download=True)
        for (image, label) in tqdm(data_set, desc="Processing data"):
            sample = {}
            sample["data"] = np.array(image)
            sample["label"] = label
            sample["class_dir"] = f"class-{label}"
            sample["training"] = train
            sample["filename"] = "None"
            data_samples += [sample]
        return data_samples

    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, data: list[dict] | tuple[dict]) -> None:
            self.data = list(data)

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            img = self.data[idx]["data"]
            img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()
            img = torch.nn.functional.interpolate(img, 56)[0]
            label = int(self.data[idx]["label"])
            return img, label

mnsit_det = MNISTDetector()
mnsit_det.load_data(labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    data_frac=1, minmax=[0, 255], scale=True)

img, label = mnsit_det.class_top.dataset_fn(mnsit_det.data)[0]
print(label)
plt.imshow(img.numpy()[0])
```
    Loading processed data..: 100%|██████████| 70000/70000 [03:59<00:00, 292.55it/s]
    Normalizing Data..: 100%|██████████| 70000/70000 [00:00<00:00, 86730.60it/s] 
    5
![An image of a handwritten numeral 5](docs/_static/dsoutput.png)

Now the [train_nn()](qgain.detector.Detector.train_nn) function will need to be overidden slightly to take advantage of the key we added to make a distinction between training and testing data. By default Q-GAIN uses the data loaded into the detector for creating the training and validation sets, but both training and testing data will be loaded into the detector. So we'll check for this flag and call the Q-GAIN functionality, with the argument *data* changed from None, on the training dataset to create the training and validation subsets to be used during training.

During training Q-GAIN will need an objective function to determine a loss and optimize the model weights. This will need to be specified here as well, so the one used by the SolDet subpackage will be provided to the argument *cl_loss_fn*.

```python
import numpy as np
import torch
from qgain.detector import Detector
from torchvision.datasets import MNIST
from tqdm import tqdm

class MNISTDetector(Detector):
    def __init__(self) -> None:
        super().__init__(process_fn=self.mnist_process_fn,
                        cl_model=qgain.soldet.classifier_nn.MLST2021CNNmodern,
                        cl_kwargs={"num_classes": 10},
                        cl_dataset_fn=self.MNISTDataset,
                        augment=None,
                        cl_loss_fn=torch.nn.NLLLoss)

    def mnist_process_fn(self, input_data: MNIST, *, train: bool = True) -> list[dict]:
        data_samples = []
        data_set = MNIST(input_data, train=train, download=True)
        for (image, label) in tqdm(data_set, desc="Processing data"):
            sample = {}
            sample["data"] = np.array(image)
            sample["label"] = label
            sample["class_dir"] = f"class-{label}"
            sample["training"] = train
            sample["filename"] = "None"
            data_samples += [sample]
        return data_samples

    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, data: list[dict] | tuple[dict]) -> None:
            self.data = list(data)

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            img = self.data[idx]["data"]
            img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()
            img = torch.nn.functional.interpolate(img, 56)[0]
            label = int(self.data[idx]["label"])
            return img, label

    def train_nn(self, patience: int = 30, epochs: int = 30) -> None:
        tr_idx = []
        for idx, item in enumerate(self.data):
            if item["training"]:
                tr_idx.append(idx)

        tr_set = map(self.data.__getitem__, tr_idx)

        super().train_nn(model_list=["classifier"], patience=patience, 
                         epochs=epochs, data=list(tr_set))
    
mnsit_det.train_nn()
```
    Device: cuda | Epoch: 30/30 | Loss: 0.036941 | Test Loss: 0.062874 | Acc.: 0.984043: 100%|██████████| 30/30 [10:15<00:00, 20.51s/it]
    Done! Minimum Test Loss: 0.052880093455314636 with Accuracy: 0.984375.

The resulting weights are saved to the models folder of the experiment with the current date and time, ending in "_classifier.pt". We can use these weights to apply our trained model to all of the data loaded into the detector. This is done by calling [use_models()](qgain.detector.Detector.use_models) and listing the weights file in the *model_paths* argument.

```python
mnsit_det.use_models(model_list=['classifier'], model_paths=['20250514_140413_classifier.pt'])
```
    Loaded /Users/qgain/MNIST/models/20250514_140413_classifier.pt.
    Starting ML Classifier.
    Classifier model loaded.
    Running model, please wait..
    Running..: 100%|██████████| 70000/70000 [03:50<00:00, 303.28it/s]
    Finished.

The results can be shown in a truth table by making use of a built in convenience function that plots the labels in the loaded dataset. By calling [plot_metrics()](qgain.detector.Detector.plot_metrics) and choosing the relevant model the function will output various plots. These plots require the labels and structure Q-GAIN expects. By choosing 'classifier' we will get a truth table, whose code will look for ground labels via the 'label' key and the predicted values by the classifier via the 'CL_pred' key. See the documentation for [plot_metrics()](qgain.detector.Detector.plot_metrics) for more information. We will again override this function so we can only select the testing data by checking the 'training' key for False.

```python
import numpy as np
import torch
from qgain.detector import Detector
from torchvision.datasets import MNIST
from tqdm import tqdm

class MNISTDetector(Detector):
    def __init__(self) -> None:
        super().__init__(process_fn=self.mnist_process_fn,
                        cl_model=qgain.soldet.classifier_nn.MLST2021CNNmodern,
                        cl_kwargs={"num_classes": 10},
                        cl_dataset_fn=self.MNISTDataset,
                        augment=None,
                        cl_loss_fn=torch.nn.NLLLoss)

    def mnist_process_fn(self, input_data: MNIST, *, train: bool = True) -> list[dict]:
        data_samples = []
        data_set = MNIST(input_data, train=train, download=True)
        for (image, label) in tqdm(data_set, desc="Processing data"):
            sample = {}
            sample["data"] = np.array(image)
            sample["label"] = label
            sample["class_dir"] = f"class-{label}"
            sample["training"] = train
            sample["filename"] = "None"
            data_samples += [sample]
        return data_samples

    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, data: list[dict] | tuple[dict]) -> None:
            self.data = list(data)

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            img = self.data[idx]["data"]
            img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()
            img = torch.nn.functional.interpolate(img, 56)[0]
            label = int(self.data[idx]["label"])
            return img, label

    def train_nn(self, patience: int = 30, epochs: int = 30) -> None:
        tr_idx = []
        for idx, item in enumerate(self.data):
            if item["training"]:
                tr_idx.append(idx)

        tr_set = map(self.data.__getitem__, tr_idx)

        super().train_nn(model_list=["classifier"], patience=patience, 
                         epochs=epochs, data=list(tr_set))
    
    def plot_metrics(self, **kwargs: dict) -> None:
        val_idx = []
        for idx, item in enumerate(self.data):
            if not item["training"]:
                val_idx.append(idx)

        val_set = map(self.data.__getitem__, val_idx)

        super().plot_metrics(types=["classifier"], data=list(val_set), **kwargs)
```

Now if we call this we should get the results of our model in table form. Q-GAIN includes optional arguments to influence the style of the output using matplotlib style sheets through the argument *style*.

```python
import os
import sys
import matplotlib.pyplot as plt

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tools import jqi_cols

plt.colormaps.register(cmap=jqi_cols, name="jqi_cols")
style_sheet = 'MNIST_ex_style'
mnsit_det.plot_metrics(**{'style': style_sheet})
```
![A truth table](docs/_static/taboutput.png)
