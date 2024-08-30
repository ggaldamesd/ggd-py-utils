# ggd_py_utils

Un conjunto de utilidades para proyectos de Python.

## Contenido del Paquete

El paquete `ggd_py_utils` incluye las siguientes utilidades:

* **Tracing**: funciones para medir el tiempo de ejecución de bloques de código.
* **Machine Learning**: utilidades para preparar datos y entrenar modelos de clasificación supervisada con FastText.
* **Formating**: funciones para formatear números y texto.

## Estructura del Paquete

El paquete se organiza de la siguiente manera:

* `ggd_py_utils/`: directorio principal del paquete
	+ `machine_learning/`: utilidades para machine learning
		- `fasttext/`: utilidades para FastText
			- `supervised/`: utilidades para clasificación supervisada
				- `training.py`: script para entrenar modelos
				- `data_preparation.py`: script para preparar datos
	+ `tracing/`: utilidades para tracing
		- `metrics.py`: script para medir el tiempo de ejecución
	+ `formating/`: utilidades para formatear números y texto
		- `numeric.py`: script para formatear números
	+ `README.md`: este archivo de README

## Instalación

Para instalar el paquete, puedes ejecutar el siguiente comando:
```bash
pip uninstall ggd-py-utils; pip install git+https://github.com/ggaldamesd/ggd-py-utils.git
```
## Requisitos

El paquete requiere las siguientes dependencias:

* `unidecode==1.3.8`
* `nltk==3.8.1`
* `pandas==2.2.2`
* `tqdm==4.66.2`
* `imbalanced-learn==0.12.3`
* `scikit-learn==1.5.0`
