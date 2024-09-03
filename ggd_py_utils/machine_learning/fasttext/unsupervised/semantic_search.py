from pandas import DataFrame
from fasttext.FastText import _FastText

from ggd_py_utils.tracing.metrics import time_block

def load_resources(
        fasttext_model_path:str,
        faiss_index_file:str,
        df_data_file:str,
    ) -> tuple[_FastText, object, DataFrame]:
    
    """
    Load the resources needed for semantic search.

    Parameters
    ----------
    fasttext_model_path : str
        The path to the FastText model.
    faiss_index_file : str
        The path to the Faiss index file.
    df_data_file : str
        The path to the DataFrame file.

    Returns
    -------
    tuple
        A tuple with the following elements:

        - model: The FastText model.
        - faiss_index: The Faiss index.
        - df: The DataFrame containing the data.
    """
    
    with time_block(block_name="Load FastText model"):
        from fasttext import load_model
        
        model:_FastText = load_model(path=fasttext_model_path)
    
    with time_block(block_name="Load Faiss index"):
        from faiss import read_index
        faiss_index = read_index(faiss_index_file)
    
    with time_block(block_name="Load DataFrame"):
        from pandas import read_pickle, DataFrame
        df:DataFrame = read_pickle(filepath_or_buffer=df_data_file)
    
    return model, faiss_index, df

def semantic_search(
        query:str, 
        model:_FastText, 
        index:object, 
        df:DataFrame,
        k:int=3,
        fields_to_project: list[str] = None
    ):
    """
    Perform a semantic search using a FastText model.

    Parameters
    ----------
    query : str
        The query to search for.
    model : _FastText
        The FastText model to use for the search.
    index : object
        The Faiss index to use for the search.
    df : DataFrame
        The DataFrame containing the data to search.
    k : int, optional
        The number of results to return, by default 3.
    fields_to_project : list[str], optional
        The list of fields to project, by default None.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary contains the columns of the DataFrame plus a 'score' column, which is the similarity score of the result to the query.
    """
    
    from ggd_py_utils.machine_learning.data.cleaning import clean_text
    
    query_cleaned:str = clean_text(text=query)
    query_vector:list = model.get_sentence_vector(text=query_cleaned)
    query_vector = query_vector.astype('float32').reshape(1, -1)
    
    D, I = index.search(query_vector, k)
    
    results = df.iloc[I[0]].copy()
    results['Similarity'] = D[0]

    if fields_to_project is not None:
        if 'Similarity' not in fields_to_project:
            fields_to_project.append('Similarity')
        results = results[fields_to_project]

    _results = results.to_dict('records') 

    return _results