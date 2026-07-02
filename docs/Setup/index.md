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