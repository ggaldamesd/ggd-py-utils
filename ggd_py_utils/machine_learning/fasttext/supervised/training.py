from fasttext.FastText import _FastText
from fasttext import train_supervised
from multiprocessing import cpu_count

from ggd_py_utils.tracing.file import get_file_size
from ggd_py_utils.tracing.metrics import time_block

def merge_hyperparameters(defaults:dict, overrides:dict={}) -> dict:
    """
    Merge two dictionaries of hyperparameters. If a key is present in both dictionaries, the value from the overrides dictionary will be used.

    Parameters
    ----------
    defaults : dict
        A dictionary of default hyperparameters.
    overrides : dict
        A dictionary of hyperparameters to override the defaults.

    Returns
    -------
    dict
        A dictionary of hyperparameters with the overrides applied.
    """
    hyperparameters:dict = {key: overrides.get(key, default) for key, default in defaults.items()}
    
    return hyperparameters

def train_fasttext_model_with_autotune(
        train_corpus_path:str, 
        validation_corpus_path:str, 
        model_file:str, 
        threads:int=cpu_count(),
        autotune_duration_in_minutes:int=5,
        autotune_model_size_in_mb:int=250,
) -> _FastText:
    
    
    """
    Train a FastText supervised model using autotune. The model is trained using the given training corpus and validated using the given validation corpus. The model is then saved to the given file.

    Parameters
    ----------
    train_corpus_path : str
        The path to the training corpus.
    validation_corpus_path : str
        The path to the validation corpus.
    model_file : str
        The path to save the model.
    threads : int, optional
        The number of threads to use, by default the number of CPUs.
    autotune_duration_in_minutes : int, optional
        The duration of the autotune in minutes, by default 5.
    autotune_model_size_in_mb : int, optional
        The size of the model in megabytes, by default 250.

    Returns
    -------
    _FastText
        The trained model.

    Notes
    -----
    The model is trained using the following hyperparameters:

    - `verbose`: 2
    - `thread`: The number of threads given
    - `autotuneValidationFile`: The validation file given
    - `autotuneMetric`: 'f1'
    - `autotunePredictions`: 1
    - `autotuneDuration`: The duration given in seconds
    - `autotuneModelSize`: The size given in bytes
    """
    model:_FastText
    
    autotune_duration:int = 60 * autotune_duration_in_minutes
    autotune_model_size:int = 1024 * 1024 * autotune_model_size_in_mb

    with time_block(block_name="Training with autotune"):
        model:_FastText = train_supervised(
            input=train_corpus_path,
            verbose=2,
            thread=threads,
            autotuneValidationFile=validation_corpus_path,
            autotuneMetric="f1",
            autotunePredictions=1,
            autotuneDuration=autotune_duration,
            autotuneModelSize=autotune_model_size
        )
    
    with time_block(block_name="Getting model metrics"):
        metrics = model.test(path=validation_corpus_path, threshold=0.0)

    print(f"Model samples: {metrics[0]}")
    print(f"Model precision: {metrics[1]}")
    print(f"Model recall: {metrics[2]}")

    model_dimensions = model.get_dimension()
    print(f"Model dimensions: {model_dimensions}")
    
    model.save_model(model_file)

    _, model_size = get_file_size(filename=model_file)

    print(f"Model size: {model_size}")
    
    return model

def train_fasttext_model_with_hyperparameter(
        train_corpus_path:str, 
        validation_corpus_path:str, 
        model_file:str, 
        threads:int=cpu_count(),
        hyperparameters:dict={},
) -> _FastText:
    """
    Train a FastText model using the given hyperparameters.

    Parameters
    ----------
    train_corpus_path : str
        The path to the training corpus file.
    validation_corpus_path : str
        The path to the validation corpus file.
    model_file : str
        The path to save the model file.
    threads : int, optional
        The number of threads to use, by default the number of available CPU cores.
    hyperparameters : dict, optional
        The hyperparameters to use, by default an empty dictionary.

    Returns
    -------
    _FastText
        The trained model.

    Notes
    -----
    The model is trained using the following default hyperparameters:
    
    - `lr`: 0.05
    - `dim`: 100
    - `ws`: 5
    - `epoch`: 5
    - `minCount`: 5
    - `minn`: 3
    - `maxn`: 6
    - `wordNgrams`: 1
    - `loss`: "softmax"
    - `bucket`: 2000000
    - `lrUpdateRate`: 100
    - `verbose`: 2
    - `pretrainedVectors`: ""
    """
    model:_FastText
    
    supervised_default:dict = {
        'model': "supervised",
        'lr': 0.05,
        'dim': 100,
        'ws': 5,
        'epoch': 5,
        'minCount': 5,
        'minn': 3,
        'maxn': 6,
        'wordNgrams': 1,
        'loss': "softmax",
        'bucket': 2000000,
        'lrUpdateRate': 100,
        'verbose': 2,
        'pretrainedVectors': "",
    }

    final_hyperparameters:dict = merge_hyperparameters(supervised_default, hyperparameters)

    with time_block(block_name="Training without autotune"):
        model:_FastText = train_supervised(
            input=train_corpus_path,
            thread=threads,
            **final_hyperparameters
        )
        
    if not model.is_quantized(): 
        with time_block(block_name="Quantizing model"):
            model.quantize()

    with time_block(block_name="Getting model metrics"):
        metrics = model.test(path=validation_corpus_path, threshold=0.0)

    print(f"Model samples: {metrics[0]}")
    print(f"Model precision: {metrics[1]}")
    print(f"Model recall: {metrics[2]}")

    model_dimensions = model.get_dimension()
    print(f"Model dimensions: {model_dimensions}")
    
    model.save_model(model_file)

    _, model_size = get_file_size(filename=model_file)

    print(f"Model size: {model_size}")
    
    return model