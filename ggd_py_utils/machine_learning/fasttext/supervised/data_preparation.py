from pandas import DataFrame, Series, to_numeric, concat
from tqdm.notebook import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from ggd_py_utils.machine_learning.data.cleaning import clean_text

tqdm.pandas()

def clean_dataframe(df:DataFrame, fields:list, inplace:bool=True) -> DataFrame:
    df.dropna(subset=fields, inplace=inplace)
    df.fillna(value="", inplace=inplace)
    
    return df

def text_label_to_numeric(df:DataFrame, label_code_name:str) -> DataFrame:
    df[label_code_name] = to_numeric(df[label_code_name], errors='coerce').astype('Int64')
    
    return df

def label_concatenation(df:DataFrame, label_fields:list) -> DataFrame:
    df["Label"] = df[label_fields].astype(str).agg('|'.join, axis=1)
    
    return df

def features_concatenation(df:DataFrame, features_fields:list):
    df["Features"] = df[features_fields].astype(str).agg(' '.join, axis=1)
    
    return df

def clean_features(df:DataFrame) -> DataFrame:
    df["Features"] = df["Features"].progress_apply(func=lambda x: clean_text(text=x))
    
    return df
    
def drop_invalid_data(df:DataFrame, label_code_name:str) -> DataFrame:
    df:DataFrame = df.query(expr=f"{label_code_name} != 0")
    
    return df

def drop_repeated_features(df:DataFrame, inplace:bool=True):
    df.drop_duplicates(subset="Features", inplace=inplace)
    
    return df

def drop_invalid_features_data(df:DataFrame) -> DataFrame:
    df:DataFrame = df[~df['Features'].str.contains(pat='prueba', case=False, na=False)]
    df:DataFrame = df[~df['Features'].str.contains(pat='LINEA INTEGRADA', case=False, na=False)]
    
    return df

def drop_numeric_encoded_labels(df:DataFrame) -> DataFrame:
    df:DataFrame = df[~df['Label'].str.contains('<NA>', na=False)]
    
    return df

def get_minimal_corpus_dataframe(df:DataFrame):
    df:DataFrame = df[["Label", "Features"]]
    
    return df

def balance_training_data(df:DataFrame) -> DataFrame:
    X: DataFrame = df.drop("Label", axis=1)
    y: Series = df["Label"].copy()
    ros = RandomOverSampler(sampling_strategy="all", random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)
    trainingData: DataFrame = concat(objs=[X_ros, y_ros], axis=1)

    return trainingData

def split_train_test_data(df:DataFrame, random_state:int=7, shuffle:bool=True, stratify:str=None, with_validation:bool=False) -> tuple:
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.2, random_state=random_state, shuffle=shuffle, stratify=strat)

    if with_validation == False:
        return (train_set, test_set)

    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=random_state, shuffle=shuffle, stratify=strat)
    
    return (train_set, val_set, test_set)

def generate_fasttext_label(text:str) -> str:
    text = "___".join(text.split())

    return text

def format_fasttext(df:DataFrame, feature_name:str="Features", label_name:str="Label", path:str="") -> DataFrame:
    df = df[[label_name, feature_name]].copy()
    df[label_name] = df[label_name].apply(func=lambda x: "__label__" + generate_fasttext_label(text=x))
    
    if len(path) != 0:
        df.to_csv(path_or_buf=path, index=False, header=False, sep="\t")

    return df