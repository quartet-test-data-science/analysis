# Analysis

A toolbox to perform s3 operations and general exploratory analyses

## Dependencies

Qml uses poetry. To get started with poetry, issue the command:

`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

To create an environment:

`make env`

To install the environment:

`make install`

To update the environment solver:

`make update-deps`

Additions can be made to the poetry environment in the normal way (with `poetry add X`).

## Instructions

To start a Jupyter notebook server, run:

`make notebook`

You should make sure that you have `JUPYTER_PORT` set in your shell rc file, along with `ANALYTICS`
