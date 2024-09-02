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

from multiprocessing import cpu_count
from fasttext.FastText import _FastText

def train_fasttext_model_with_hyperparameter(
        train_corpus_path:str, 
        model_file:str, 
        threads:int=cpu_count(),
        hyperparameters:dict={},
) -> _FastText:
    """
    Train a FastText unsupervised model using the given hyperparameters.

    Parameters
    ----------
    train_corpus_path : str
        The path to the training corpus file.
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

    - `model`: "skipgram"
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
    
    unsupervised_default:dict = {
        'model': "skipgram",
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

    final_hyperparameters:dict = merge_hyperparameters(unsupervised_default, hyperparameters)

    from ggd_py_utils.tracing.metrics import time_block
    
    with time_block(block_name="Training unsupervised model"):
        from fasttext.FastText import train_unsupervised
        
        model:_FastText = train_unsupervised(
            input=train_corpus_path,
            thread=threads,
            **final_hyperparameters
        )
    
    with time_block(block_name="Getting model metrics"):
        analogies = model.get_analogies(wordA="clavo", wordB="martillo", wordC="lechuga", k=3)

        print(f"Model metrics (analogies => clavo, martillo, lechuga): {vars(analogies)}")
        
        word_a_vector = model.get_word_vector(word="clavo")
        word_b_vector = model.get_word_vector(word="martillo")

        from numpy.linalg import norm
        
        similarity = word_a_vector.dot(word_b_vector) / (norm(word_a_vector) * norm(word_b_vector))
        print(f"Similarity between 'clavo' and 'martillo': {similarity}")

        from ggd_py_utils.machine_learning.fasttext.unsupervised.predictions import print_nearest_neighbors
        
        print_nearest_neighbors(products=["clavo", "martillo", "lechuga"], model=model)
        
    model_dimensions = model.get_dimension()
    print(f"Model dimensions: {model_dimensions}")
    
    model.save_model(model_file)

    from ggd_py_utils.tracing.file import get_file_size
    
    _, model_size = get_file_size(filename=model_file)

    print(f"Model size: {model_size}")
    
    return model

def reduce_fasttext_model(model:_FastText, target_dim:int, reduced_model_file:str) -> _FastText:
    """
    Reduce a FastText model to a lower dimension.

    Parameters
    ----------
    model : _FastText
        The model to reduce.
    target_dim : int
        The target dimension to reduce to.
    reduced_model_file : str
        The path to save the reduced model to.

    Returns
    -------
    _FastText
        The reduced model.
    """
    print(f"Target dimension: {target_dim}")

    from fasttext.util import reduce_model
    
    reduced_model: _FastText = reduce_model(ft_model=model, target_dim=target_dim)
    model.save_model(path=reduced_model_file)
    
    return reduced_model