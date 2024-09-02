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

def text_label_to_numeric(df:DataFrame, label_code_name:str) -> DataFrame:
    """
    Convert a text column to a numeric column using pandas.to_numeric.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    label_code_name : str
        The name of the column to convert.
    
    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    from pandas import to_numeric

    df[label_code_name] = to_numeric(df[label_code_name], errors='coerce').astype('Int64')
    
    return df

def label_concatenation(df:DataFrame, label_fields:list) -> DataFrame:
    """
    Concatenate multiple label columns into one column using the '|' separator.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    label_fields : list
        The column names to concatenate.
    
    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    df["Label"] = df[label_fields].astype(str).agg('|'.join, axis=1)
    
    return df

def features_concatenation(df:DataFrame, features_fields:list):
    """
    Concatenate multiple feature columns into one column using the ' ' separator.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    features_fields : list
        The column names to concatenate.
    
    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    df["Features"] = df[features_fields].astype(str).agg(' '.join, axis=1)
    
    return df

def clean_features(df:DataFrame) -> DataFrame:
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
    
    df["Features"] = df["Features"].apply(func=lambda x: clean_text(text=x))
    
    return df
    
def drop_invalid_data(df:DataFrame, label_code_name:str) -> DataFrame:
    """
    Drop all rows in the DataFrame where the value in the column named
    label_code_name is 0.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.
    label_code_name : str
        The column name to query.

    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    
    df:DataFrame = df.query(expr=f"{label_code_name} != 0")
    
    return df

def drop_repeated_features(df:DataFrame, inplace:bool=True):
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
    
    df.drop_duplicates(subset="Features", inplace=inplace)
    
    return df

def drop_invalid_features_data(df:DataFrame) -> DataFrame:
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

    df:DataFrame = df[~df['Features'].str.contains(pat='prueba', case=False, na=False)]
    df:DataFrame = df[~df['Features'].str.contains(pat='LINEA INTEGRADA', case=False, na=False)]
    
    return df

def drop_numeric_encoded_labels(df:DataFrame) -> DataFrame:
    """Drop all rows in the DataFrame where the value in the "Label" column
    contains the string '<NA>' (case sensitive).

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify.

    Returns
    -------
    DataFrame
        The modified DataFrame.
    """
    df:DataFrame = df[~df['Label'].str.contains('<NA>', na=False)]
    
    return df

def get_minimal_corpus_dataframe(df:DataFrame):
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
    df:DataFrame = df[["Label", "Features"]]
    
    return df

def balance_training_data(df:DataFrame, method:str="oversampling", min_samples:int=10) -> DataFrame:
    """
    Balance the training data using different techniques based on the specified method.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to balance.
    method : str, optional
        The method to use for balancing the data. Options are:
        - "oversampling": Use RandomOverSampler.
        - "undersampling": Use RandomUnderSampler.
        - "smote": Use SMOTE.
    min_samples : int, optional
        The minimum number of samples required for each class. Defaults to 10.
        
    Returns
    -------
    DataFrame
        The balanced DataFrame.
    """
    from pandas import Series
    
    class_counts: Series[int] = df['Label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered: DataFrame = df[df['Label'].isin(valid_classes)]

    if method == "oversampling":
        from imblearn.over_sampling import RandomOverSampler
        
        ros = RandomOverSampler(sampling_strategy="auto", random_state=42)

        X: DataFrame = df_filtered.drop("Label", axis=1)
        y: Series = df_filtered["Label"].copy()        

        X_balanced, y_balanced = ros.fit_resample(X, y)
        
        from pandas import concat
        trainingData: DataFrame = concat(objs=[X_balanced, y_balanced], axis=1)
        
        return trainingData
    
    elif method == "undersampling":
        from imblearn.under_sampling import RandomUnderSampler
        
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
        
        X: DataFrame = df_filtered.drop("Label", axis=1)
        y: Series = df_filtered["Label"].copy()

        X_balanced, y_balanced = rus.fit_resample(X, y)
    
        from pandas import concat
        trainingData: DataFrame = concat(objs=[X_balanced, y_balanced], axis=1)
        
        return trainingData
    elif method == "smote":
        from sklearn.preprocessing import LabelEncoder
        
        X = df_filtered['Features']
        y = df_filtered['Label']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    
        from sklearn.model_selection import train_test_split
    
        X_train, X_test, y_train, _ = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

        from sklearn.feature_extraction.text import TfidfVectorizer
            
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        
        min_samples_in_class = min([y_train.tolist().count(label) for label in set(y_train)])

        k_neighbors:int = min(5, min_samples_in_class - 1) if min_samples_in_class > 1 else 1
        
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)

        X_train_vec_dense = X_train_vec.toarray()
        X_resampled, y_resampled = smote.fit_resample(X_train_vec_dense, y_train)
        
        X_resampled_df = DataFrame(X_resampled, columns=vectorizer.get_feature_names_out())

        y_balanced = le.inverse_transform(y_resampled)

        X_resampled_df['Label'] = y_balanced
        
        X_resampled_df:DataFrame = X_resampled_df[['Label'] + [col for col in X_resampled_df.columns if col != 'Label']]

        X_train_df = DataFrame(X_train)
        X_train_df['Label'] = y_train

        training_data = X_resampled_df.reset_index(drop=True)
        training_data['Features'] = X_train_df['Features'].reset_index(drop=True)
        
        training_data:DataFrame = training_data[["Label", "Features"]]

        return training_data
        
    else:
        raise ValueError("Invalid method. Choose from 'oversampling', 'undersampling' or 'smote'.")

def split_train_test_data(df:DataFrame, random_state:int=7, shuffle:bool=True, stratify:str=None, with_validation:bool=False) -> tuple:
    """
    Split the given DataFrame into training and test sets, optionally with a validation set.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to split.
    random_state : int, optional
        The random state to use for the split, by default 7.
    shuffle : bool, optional
        Whether to shuffle the data before splitting, by default True.
    stratify : str, optional
        The column to use for stratifying the split, by default None.
    with_validation : bool, optional
        Whether to return a validation set, by default False.

    Returns
    -------
    tuple
        A tuple of (train_set, test_set) or (train_set, val_set, test_set) depending on the value of with_validation.
    """
    strat = df[stratify] if stratify else None
    
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(
        df, test_size=0.2, random_state=random_state, shuffle=shuffle, stratify=strat)

    if with_validation == False:
        return (train_set, test_set)

    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=random_state, shuffle=shuffle, stratify=strat)
    
    return (train_set, val_set, test_set)

def generate_fasttext_label(text:str) -> str:
    """
    Replace spaces in a string with '___' to make it suitable as a label in a FastText model.

    Parameters
    ----------
    text : str
        The string to process.

    Returns
    -------
    str
        The string with spaces replaced by '___'.
    """
    text = "___".join(text.split())

    return text

def format_fasttext(df:DataFrame, feature_name:str="Features", label_name:str="Label", path:str="") -> DataFrame:
    """
    Format a DataFrame to be suitable as input to a FastText supervised model.

    The DataFrame should have two columns: a label column and a feature column.
    The label column should contain the text labels for the data points.
    The feature column should contain the text features for the data points.

    The function will drop all columns except for the label and feature columns,
    apply the generate_fasttext_label function to the label column, and save the
    DataFrame to the given path if one is provided.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to format.
    feature_name : str, optional
        The name of the feature column, by default "Features".
    label_name : str, optional
        The name of the label column, by default "Label".
    path : str, optional
        The path to save the formatted DataFrame to, by default "" (do not save).

    Returns
    -------
    DataFrame
        The formatted DataFrame.
    """
    df = df[[label_name, feature_name]].copy()
    df[label_name] = df[label_name].apply(func=lambda x: "__label__" + generate_fasttext_label(text=x))
    
    if len(path) != 0:
        df.to_csv(path_or_buf=path, index=False, header=False, sep="\t")

    return df

def prepare_corpus_dataframe(
        df:DataFrame, fields_to_clean:list, 
        label_code:str, label_name:str, 
        features_fields:list, 
        corpus_ft_path:str, 
        validation_corpus_ft_path:str, 
        balance_method:str="oversampling", 
        dimensions:int=300
    ):
    """
    Prepare a DataFrame to be used as input to a FastText supervised model.

    The function will clean the DataFrame by removing rows with NaN values in the specified columns and replacing NaN values with an empty string.
    It will then convert the label column to a numeric column, concatenate the label and feature columns, clean the features column, drop repeated features, drop invalid features data, drop numeric encoded labels, balance the training data, split the data into a training set and a test set, format the data for FastText, and calculate the estimated number of parameters in the model.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to prepare.
    fields_to_clean : list
        The columns names to clean.
    label_code : str
        The column name of the label code.
    label_name : str
        The column name of the label name.
    features_fields : list
        The column names of the features.
    corpus_ft_path : str
        The path to save the formatted training data.
    validation_corpus_ft_path : str
        The path to save the formatted test data.
    balance_method : str, optional
        The method to use for balancing the data. Options are:
        - "oversampling": Use RandomOverSampler.
        - "undersampling": Use RandomUnderSampler.
        - "smote": Use SMOTE for synthetic oversampling.
        - "class_weight": Assign class weights based on class frequency (default: "oversampling").
    dimensions : int, optional
        The number of dimensions to use in the model, by default 300.

    Returns
    -------
    None
    """
    print(f"Initial Dataframe shape: {df.shape}")
    
    from ggd_py_utils.tracing.metrics import time_block
    
    with time_block(block_name="clean_dataframe"):
        df:DataFrame = clean_dataframe(df=df, fields=fields_to_clean, inplace=True)
        print(f"Dataframe shape after first clean: {df.shape}")

    with time_block(block_name="clean_dataframe"):
        df:DataFrame = text_label_to_numeric(df=df, label_code_name=label_code)

    with time_block(block_name="label_concatenation"):
        df:DataFrame = label_concatenation(df=df, label_fields=[label_code, label_name])

    with time_block(block_name="features_concatenation"):
        df:DataFrame = features_concatenation(df=df, features_fields=features_fields)

    with time_block(block_name="clean_features"):
        df:DataFrame = clean_features(df=df)

    with time_block(block_name="clean_features"):
        df:DataFrame = drop_invalid_data(df=df, label_code_name=label_code)
        print(f"Dataframe shape after fatures clean: {df.shape}")

    with time_block(block_name="drop_repeated_features"):
        df:DataFrame = drop_repeated_features(df=df)
        print(f"Dataframe shape after drop_repeated_features: {df.shape}")

    with time_block(block_name="drop_invalid_features_data"):
        df:DataFrame = drop_invalid_features_data(df=df)
        print(f"Dataframe shape after drop_invalid_features_data: {df.shape}")
        
    with time_block(block_name="drop_numeric_encoded_labels"):
        df:DataFrame = drop_numeric_encoded_labels(df=df)
        print(f"Dataframe shape after drop_numeric_encoded_labels: {df.shape}")

    with time_block(block_name="get_minimal_corpus_dataframe"):
        df:DataFrame = get_minimal_corpus_dataframe(df=df)

    with time_block(block_name="balance_training_data"):
        df:DataFrame = balance_training_data(df=df, method=balance_method)
        print(f"Dataframe shape after balance_training_data: {df.shape}")

    with time_block(block_name="split_train_test_data"):
        train, test = split_train_test_data(df=df)

        print("Training Set:", len(train))
        print("Test Set:", len(test))

    with time_block(block_name="format_fasttext_train_data"):
        format_fasttext(df=train, path=corpus_ft_path)
        
    with time_block(block_name="format_fasttext_validation_data"):
        format_fasttext(df=test, path=validation_corpus_ft_path)

    from ggd_py_utils.machine_learning.data.corpus_metrics import get_words_and_subwords_counts
    
    words_and_subwords_counts:dict = get_words_and_subwords_counts(filename=corpus_ft_path)

    words:int = words_and_subwords_counts["words"]
    subwords:int = words_and_subwords_counts["subwords"]
    tokens:int = words + subwords
    
    estimated_params:int = dimensions * tokens
    
    from ggd_py_utils.formating.numeric import abbreviate_large_number
    estimated_params_formated:str = abbreviate_large_number(number=estimated_params)

    print(f"Estimated corpus parameters: {estimated_params} / {estimated_params_formated}")