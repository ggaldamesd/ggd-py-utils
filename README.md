**Package Purpose**

The `ggd-py-utils` package is a collection of reusable utilities for Python projects. The package is designed to provide a set of useful tools for various tasks, including tracing, machine learning, and formatting.

**Package Structure**

The package is organized into the following directories and modules:

* `machine_learning/`: utilities for machine learning
	+ `fasttext/`: utilities for FastText
		- `supervised/`: utilities for supervised classification
			- `training.py`: script for training models
			- `data_preparation.py`: script for preparing data
* `tracing/`: utilities for tracing
	+ `metrics.py`: script for measuring execution time
* `formating/`: utilities for formatting numbers and text
	+ `numeric.py`: script for formatting numbers

**Package Contents**

The package includes the following files and directories:

* `ggd_py_utils/`: package directory
	+ `machine_learning/`: machine learning utilities
	+ `tracing/`: tracing utilities
	+ `formating/`: formatting utilities
	+ `README.md`: this README file

**Dependencies**

The package requires the following dependencies:

* `unidecode==1.3.8`
* `nltk==3.8.1`
* `pandas==2.2.2`
* `tqdm==4.66.2`
* `imbalanced-learn==0.12.3`
* `scikit-learn==1.5.0`

**Installation**

To install the package, run the following command:
```bash
pip uninstall ggd-py-utils; pip install git+https://github.com/ggaldamesd/ggd-py-utils.git
```
**Updating**

To update the package, run the following command:
```bash
pip install --upgrade git+https://github.com/ggaldamesd/ggd-py-utils.git
```
**Uninstallation**

To uninstall the package, run the following command:
```bash
pip uninstall ggd-py-utils
```
