# iris-classification

https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

## Pre-requisites

1. Install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html):

    ```bash
    pip3 install virtualenvwrapper
    ```

1. Create virtual environment in `~/.virtualenvs`:

    ```bash
    mkvirtualenv iris-classification
    ```

1. Install packages:

    ```bash
    pip install --requirement requirements.txt
    ```

## Run

1. Activate virtual environment:

    ```bash
    workon iris-classification
    ```

1. Toggle the outputs by commenting/un-commenting the printouts at the bottom of `iris.py`, then run:

    ```python
    python iris.py
    ```
