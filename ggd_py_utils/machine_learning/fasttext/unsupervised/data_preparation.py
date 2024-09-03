from pandas import DataFrame

def clean_dataframe(df:DataFrame, fields:list, inplace:bool=True) -> DataFrame:
    """
    Clean a DataFrame by removing rows with NaN values in the specified
    columns and replacing NaN values with an empty string.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to clean.
    fields : list
        The column names to clean.
    inplace : bool, optional
        Whether to modify the original DataFrame or return a copy. Defaults to True.

    Returns
    -------
    DataFrame
        The cleaned DataFrame.
    """
    df.dropna(subset=fields, inplace=inplace)
    df.fillna(value="", inplace=inplace)
    
    return df

def define_features(df:DataFrame, features_field_name:str="Features", ignore:list=[], only_return_features:bool=False) -> DataFrame:
    """
    Define a new column in the DataFrame with the given name containing the
    concatenation of all string columns in the DataFrame, separated by a space.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    features_field_name : str, optional
        The name of the new column, by default "Features".
    ignore : list, optional
        A list of column names to ignore, by default [].
    only_return_features : bool, optional
        Whether to return only the new column with the features, by default False.

    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    df[features_field_name] = df.apply(lambda row: " ".join([str(row[col]) for col in df.columns if col not in ignore and isinstance(row[col], str)]), axis=1)
    if only_return_features: df = df[[features_field_name]]
    
    return df

def clean_features(df:DataFrame, features_field_name:str="Features") -> DataFrame:
    """
    Clean the features column by applying the clean_text function to each value.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    
    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    from ggd_py_utils.machine_learning.data.cleaning import clean_text
    
    df[features_field_name] = df[features_field_name].apply(func=lambda x: clean_text(text=x))
    
    return df
    
def drop_repeated_features(df:DataFrame, features_field_name:str="Features", inplace:bool=True):
    """
    Drop all rows in the DataFrame where the value in the "Features" column
    already exists.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    inplace : bool, optional
        Whether to modify the original DataFrame or return a copy. Defaults to True.

    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    
    df.drop_duplicates(subset=features_field_name, inplace=inplace)
    
    return df

def drop_invalid_features_data(df:DataFrame, features_field_name:str="Features") -> DataFrame:
    """Drop all rows in the DataFrame where the value in the "Features" column
    contains the strings 'prueba' or 'LINEA INTEGRADA' (case insensitive).

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.

    Returns
    -------
    DataFrame
        The modified DataFrame.
    """

    df:DataFrame = df[~df[features_field_name].str.contains(pat='prueba', case=False, na=False)]
    df:DataFrame = df[~df[features_field_name].str.contains(pat='LINEA INTEGRADA', case=False, na=False)]
    
    return df

def get_minimal_corpus_dataframe(df:DataFrame, features_field_name:str="Features") -> DataFrame:
    """
    Return a DataFrame with only the "Label" and "Features" columns.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    
    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    df:DataFrame = df[[features_field_name]]
    
    return df

def format_fasttext(df:DataFrame, features_field_name:str="Features", path:str="") -> DataFrame:
    """
    Format a DataFrame to be suitable as input to a FastText supervised model.

    The DataFrame should have a single column named "Features" containing the text features for the data points.

    The function will apply the generate_fasttext_label function to the "Features" column, and optionally save the
    DataFrame to the given path if one is provided.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to format.
    features_field_name : str, optional
        The name of the feature column, by default "Features".
    path : str, optional
        The path to save the formatted DataFrame to, by default "" (do not save).

    Returns
    -------
    DataFrame
        The formatted DataFrame.
    """
    df = df.apply(func=lambda x: f"{x[features_field_name]}", axis=1)

    if len(path) != 0:
        df.to_csv(path_or_buf=path, index=False, header=False, sep="\t")

    return df

def prepare_corpus_dataframe(
        df:DataFrame, fields_to_clean:list, 
        corpus_ft_path:str, 
        features_field_name:str="Features", 
        dimensions:int=300,
        ignore_features_fields:list=[]
    ):
    print(f"Initial Dataframe shape: {df.shape}")
    
    from ggd_py_utils.tracing.metrics import time_block
    
    with time_block(block_name="clean_dataframe"):
        df:DataFrame = clean_dataframe(df=df, fields=fields_to_clean, inplace=True)
        print(f"Dataframe shape after first clean: {df.shape}")

    with time_block(block_name="define_features"):
        df:DataFrame = define_features(df=df, features_field_name=features_field_name, ignore=ignore_features_fields, only_return_features=True)

    with time_block(block_name="clean_features"):
        df:DataFrame = clean_features(df=df, features_field_name=features_field_name)

    with time_block(block_name="drop_repeated_features"):
        df:DataFrame = drop_repeated_features(df=df, features_field_name=features_field_name)
        print(f"Dataframe shape after drop_repeated_features: {df.shape}")

    with time_block(block_name="drop_invalid_features_data"):
        df:DataFrame = drop_invalid_features_data(df=df, features_field_name=features_field_name)
        print(f"Dataframe shape after drop_invalid_features_data: {df.shape}")
        
    with time_block(block_name="get_minimal_corpus_dataframe"):
        df:DataFrame = get_minimal_corpus_dataframe(df=df, features_field_name=features_field_name)

    with time_block(block_name="format_fasttext_train_data"):
        format_fasttext(df=df, path=corpus_ft_path, features_field_name=features_field_name)
        
    from ggd_py_utils.machine_learning.data.corpus_metrics import get_words_and_subwords_counts
    
    words_and_subwords_counts:dict = get_words_and_subwords_counts(filename=corpus_ft_path)

    words:int = words_and_subwords_counts["words"]
    subwords:int = words_and_subwords_counts["subwords"]
    tokens:int = words + subwords
    
    estimated_params:int = dimensions * tokens
    
    from ggd_py_utils.formating.numeric import abbreviate_large_number
    estimated_params_formated:str = abbreviate_large_number(number=estimated_params)

    print(f"Estimated corpus parameters: {estimated_params} / {estimated_params_formated}")