from fasttext.FastText import _FastText
from fasttext import train_supervised
from multiprocessing import cpu_count

from ggd_py_utils.tracing.metrics import time_block

def train_fasttext_model(
        train_corpus_path:str, 
        validation_corpus_path:str, 
        model_file:str, 
        auto_tune_model:bool=False, 
        threads:int=cpu_count(),
        autotune_duration_in_minutes:int=5,
        autotune_model_size_in_mb:int=250,
) -> _FastText:
    """
    Train a FastText supervised model.

    Parameters
    ----------
    train_corpus_path : str
        The path to the file containing the training corpus.
    validation_corpus_path : str
        The path to the file containing the validation corpus.
    model_file : str
        The path to save the trained model.
    auto_tune_model : bool, optional
        If True, the hyperparameters of the model will be tuned using the autotune feature of the FastText library, by default False.
    threads : int, optional
        The number of threads to use during training, by default the number of CPUs available.
    autotune_duration_in_minutes : int, optional
        The duration of the autotuning process in minutes, by default 5.
    autotune_model_size_in_mb : int, optional
        The size of the model in megabytes, by default 250.

    Returns
    -------
    _FastText
        The trained FastText model.
    """

    model:_FastText
    
    if auto_tune_model:
        autotune_duration:int = 60 * autotune_duration_in_minutes
        autotune_model_size:int = 1024 * 1024 * autotune_model_size_in_mb

        with time_block(block_name="Training with autotune"):
            model:_FastText = train_supervised(
                input=train_corpus_path,
                verbose=3,
                thread=threads,
                autotuneValidationFile=validation_corpus_path,
                autotuneMetric="f1",
                autotunePredictions=1,
                autotuneDuration=autotune_duration,
                autotuneModelSize=autotune_model_size
            )
    else:
        with time_block(block_name="Training without autotune"):
            model:_FastText = train_supervised(
                input=train_corpus_path,
                verbose=3,
                thread=threads
            )

    with time_block(block_name="Getting model metrics"):
        metrics = model.test(path=validation_corpus_path, threshold=0.0)

    print(f"Número de ejemplos: {metrics[0]}")
    print(f"Precisión: {metrics[1]}")
    print(f"Recall: {metrics[2]}")

    model.save_model(model_file)
    return model