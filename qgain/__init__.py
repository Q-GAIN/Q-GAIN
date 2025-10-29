"""Package initialization.

Note that if no config file is found it will be created with defaults during import.
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
