# Running Features of Q-GAIN

## Training ML Models

To train the ML models you can use the Detector function train_nn(). The argument *model_list* specifies which models to train on the data. This expects a list of the available models, 'classifier' or 'object detector'. The arguments *epochs* and *patience* control how long to train the models. The trained weights will be saved to the models folder in your experiment directory. Other arguments and their descriptions can be found in the API reference for the function.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(tags=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.train_nn(model_list=['classifier', 'object detector'], patience=10, epochs=50)
```

The default state of Q-GAIN is to use the data loaded into the current detector object. However, if you require a different set of data to train from you can change the argument *data* from None to the desired set of data. Note that this list or dictionary of samples must have the structure anticipated by Q-GAIN.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(tags=[0, 1], data_frac=0.9, minmax=[-1, 3])
tr_set = []
for item in sd.data:
    if item["label"] == 1:
        tr_set.append({"data":item["data"], "label":item["label"], "positions":item["positions"]})

sd.train_models(model_list=["classifier", "object detector"], patience=10, epochs=50, data=tr_set)

```

## Building Statistical Tools

These analysis algorithms sometimes require fitting or calculating new paramaters. This fitting procedure can be started by invoking the detector method *define_stat*, which will work through every tool loaded into the statistic controller. You can specify specific tools by passing their names to the *tool_list* argument.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(tags=[0, 1], data_frac=0.9, minmax=[-1, 3])
sd.define_stat(tool_list=["pie classifier"], save=True)
```

Just as in the ML case, this function can also fit to external data rather than any internally loaded data by passing it to the *data* argument.

## Making Use of The Analysis Tools

To use any of these analysis tools you can make use of the use_models() function of a detector. Specific tools can be specified with the *model_list* argument. If none are given then the detector will attempt to use all available tools. The argument *model_paths* can be used to dictate any saved ML weight files or statistical tool parameter states located in the models folder of the experiment path. These files should end in the name of the tool being used. Results are saved in the data dictionary for each sample.

```python
from qgain.soldet import soliton_detector

sd = soliton_detector.SolitonDetector()
sd.load_data(tags=[1], data_frac=0.9, minmax=[-1, 3])

sd.use_models(model_list=['classifier', 'object detector', 'pie classifier', 'quality estimator'], 
              model_paths=['20240318_215015_CL.pt', '20240318_225331_OD.pt', 
                             '20241210_223439_pie classifier.pkl', '20241210_223508_quality estimator.pkl'])      
```
