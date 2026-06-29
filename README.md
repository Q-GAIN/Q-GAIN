Q-GAIN is a Python library serving as the foundation of a growing modular architecture to support a wider range of machine learning and physically informed analysis applications. 

It is featured with the original SolDet package as a submodule which includes classification, object detection, and physics informed metric methods. Technical details of the original package are explained in https://arxiv.org/abs/2111.04881. The dataset used to prepare the original SolDet and used by the modern SolDet module is available at https://data.nist.gov/od/id/mds2-2363. It also includes a vortex detector as a submodule for identifying the positions of vortices in Bose-Einstein Condensates.

# Installing Q-GAIN

Installing Q-GAIN requires python to be >= 3.10.

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

First, clone or download the github repository and place it in your desired directory.  Assuming the usage of git you can use the git command in Python or your system environment.
```console
git clone url_to_repository
cd path_to_qgain_dir
```
Change to this directory before installing Q-GAIN. From the console invoke pip and install in editable mode with the option -e if you plan to modify the library, or without the editable flag if you with to install normally.
```console
pip install -e qgain
```

# Getting started

The main features of Q-GAIN are accessed with the `Detector` class, and some helper functions are available as part of the Q-GAIN library to assist with its usage. The intended usage of this class is to sub-class it to create new detector objects and modify it with the framework to suit your needs.

```python
class NewDetector(Detector):
    def __init__(self) -> None:
        super().__init__()
```
However, Q-GAIN comes with two detectors by default: one for solitons and one for vortices. For solitons the SolDet sub package can be used. This will create a SolitonDetector object, which inherits the functionality of Q-GAIN's Detector class. Similarly, for vortices the vdet subpackage can be used to create a VortexDetector object.

```python
from qgain.soldet import soliton_detector
from qgain.vdet import vortex_detector

sd = soliton_detector.SolitonDetector()
vd = vortex_detector.VortexDetector()
```

These can be used as is for their respective analysis types, or modified for similar analysis needs. For more information on configuring Q-GAIN please see the internal documentation or visit the Examples repository.

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

Data folders holding analysis data reside within the <em>data_files</em> directory. The roster file used by Q-GAIN to find these files resides within the <em>data_info</em> directory. 

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
```