"""Q-GAIN library.

A modular architecture to support a range of machine learning and physically informed analysis applications.
The main features of Q-GAIN are accessed with the `Detector` class, and some helper functions are available as part of
the Q-GAIN library to assist with its usage. The following will be a demonstration of using the SolDet subpackage of
Q-GAIN for soliton detection.

To start, import soldet and create a SolitonDetector object, which inherits the functionality of Q-GAIN's Detector
class.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
```

On import, or when creating the Detector object, Q-GAIN will check to confirm if a configuration file,
CONFIG.ini, exists in its package path. If not this is created with default values.

[PATHS]

data_path = [path to home directory]/qgain

def_exp_name = default_ds

These set up the target directories for the required folder structure to run Q-GAIN. The <em>data_path</em> points to
the directory all experimental data folders will reside in. An experiment can be specified with <em>def_exp_name</em>,
which will set the target directory for where Q-GAIN's class data will be saved to. The default structure is,

data_path/

|- def_exp_name/

` `|- data/

` ` ` `|- data_files/

` ` ` `|- data_info/

` `|- models/

|- ...

Class folders reside within the <em>data_files</em> directory, and their naming depends on the information contained
within the roster file, residing within the <em>data_info</em> directory. The default SolDet class structure is,

data_files/

` `|- class-0/

` `|- class-1/

` `|- class-2/

` `|- class-8/

` `|- class-9/

Multiple experiment folders can reside in the data path, and any detector objects created will reference the current
def_exp_name. Changing to a different experiment folder requires creating another detector object. Although you can
modify the configuration file manually, helper functions exist to do this for you.

To change the data path you can use the utility function change_path().

```python
qgain.change_path('../SolitonDetector/')
```

To change the current experiment directory you can use change_exp().

```python
qgain.change_exp('TestExp')
```

Although by default Q-GAIN will create the necessary folder structure if it does not exist, you may also manually
trigger this creation with the use of the config() function. This will also create the CONFIG file if it is not found.

```python
EXP_PATH, EXP_NAME = qgain.config()
print(EXP_PATH)
print(EXP_NAME)
```

Package Modules
---------------
qgain.io
    The input/output functions for Q-GAIN.
qgain.run_classifier
    The classifier controller.
qgain.run_metric
    The conventional analysis controller.
qgain.run_od
    The object detection controller.
qgain.utilities
    Support functions for Q-GAIN.

Subpackage Modules
------------------
qgain.soldet
    The SolDet library for detection and tracking of solitons.
qgain.vdet
    A vortex detector for the identification of vortices.
"""
from configparser import ConfigParser
from pathlib import Path

from qgain import soldet as soldet
from qgain import vdet as vdet
from qgain.detector import Detector as Detector
from qgain.utilities import change_exp as change_exp
from qgain.utilities import change_path as change_path
from qgain.utilities import config as config

soldet_path = Path(__file__).parent
home_dir = Path.home()

if not soldet_path.joinpath("CONFIG.ini").is_file():
    configfile = ConfigParser()
    configfile["PATHS"] = {"data_path": str(home_dir.joinpath("qgain")), "def_exp_name": "default_ds"}
    with Path(soldet_path.joinpath("CONFIG.ini")).open("w", encoding="utf-8") as file:
        configfile.write(file)
