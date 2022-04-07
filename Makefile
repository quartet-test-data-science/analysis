PYTHON_EXECUTABLE ?= python3
TOPDIR = $(PWD)
JUPYTER_PORT ?= ${JUPYTER_PORT:-8888}
VALID_BUMPS = major minor patch
KERNELSPEC = analysis
BUMPS = $(addprefix bump-, $(VALID_BUMPS))

# Set up the poetry virtual environment
.PHONY: env
env:
	poetry env use $(PYTHON_EXECUTABLE)

# Install all deps and the package locally for development
.PHONY: install
install:
	poetry install

# Update the installed deps and lockfile according to the constraints in pyproject.toml
.PHONY: update-deps
update-deps:
	poetry update


# Install this poetry environment into your jupyter kernelspec
.PHONY: kernelspec
kernelspec:
	poetry run python -m ipykernel install --user --name $(KERNELSPEC) --display-name "$(KERNELSPEC)"

# Run a secure Jupyter Notebook server in the environment on $JUPYTER_PORT or if unset, port 8888
# Ensure that this port is forwarded in your host machine's ~/.ssh/config
.PHONY: notebook
notebook:
	poetry run jupyter notebook --no-browser --port $(JUPYTER_PORT)
