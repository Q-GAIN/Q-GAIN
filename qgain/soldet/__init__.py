"""SolDet subpackage for the identification and tracking of solitons.

Package Modules
---------------

classifier_nn
    The model for the SolDet classifier.
object_nn
    The model for the SolDet object detector.
mhat_metric
    Support functions and fitting for the physically informed models.
pi_models
    The methods for the PIE and QE models.
soliton_datasets
    Dataset modules used to prepare data for use by SolDet analysis modules.
soliton_detector
    The base SolitonDetector class.

"""
from qgain.soldet import classifier_nn as classifier_nn
from qgain.soldet import mhat_metric as mhat_metric
from qgain.soldet import object_nn as object_nn
from qgain.soldet import pi_models as pi_models
from qgain.soldet import soliton_datasets as soliton_datasets
from qgain.soldet import soliton_detector as soliton_detector
