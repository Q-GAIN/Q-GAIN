# Getting Started

To start, import the Q-GAIN library. Q-GAIN will check to confirm if a configuration file, CONFIG.ini, exists in its package path. If not this is created with default values. 

[PATHS]\
data_path = \<path to home directory\>/qgain\
def_exp_name = default_ds

These point to the target directories which hold the required folder structure to run Q-GAIN. The <em>data_path</em> points to the directory all experimental data folders will reside in. An experiment can be specified with <em>def_exp_name</em>, which will set the target directory for where Q-GAIN's class data will be saved to. The default structure is,

data_path\
|- def_exp_name\
    |- data\
        |- data_files\
        |- data_info\
    |- models\
|- ...

Class data folders reside within the <em>data_files</em> directory, and their naming depends on the information specified during the import process. Data location and their coresponding tag are kept within the roster file, residing within the <em>data_info</em> directory. Q-GAIN treats the tag metadata as a value which differentiates one type of data from another.

Multiple experiment folders can reside in a data path, and any detector objects created will reference the current def_exp_name. Changing to a different experiment folder requires creating another detector object. If the current directory is not found during detector instantiation then the default foldser structure is created.

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

The main features of Q-GAIN are accessed with the `Detector` class, and some helper functions are available as part of the Q-GAIN library to assist with its usage.

To start, import Q-GAIN and create a detector object.  This can be accomplished in two ways:
1) Use one of the built-in detector modules. 
2) Sub-class the base Detector class and build up new detectors for specific analysis tasks.

## Option 1
Q-GAIN comes with two built-in detectors by default: one for solitons and one for vortices. For solitons the SolDet sub package can be used. This will create a SolitonDetector object, which inherits the functionality of Q-GAIN's Detector class. Similarly, for vortices the vdet subpackage can be used to create a VortexDetector object.

```python
from qgain.soldet import soliton_detector
from qgain.vdet import vortex_detector

sd = soliton_detector.SolitonDetector()
vd = vortex_detector.VortexDetector()
```

## Option 2
To create new detectors you can sub-class the base Detector class.  

```python
class DummyDetector(Detector):
    def __init__:
        super().__init__()
```

This will immediately give access to the Q-GAIN framework which can be used to build up the detector for specific tasks. Low level functionality is handled by the library and the user only needs to write modular code for higher level functionality.

This higher level functionality can be plugged in by using the initialization arguments for the parent Detector class. Information on configuring these are given in the next modularity section.