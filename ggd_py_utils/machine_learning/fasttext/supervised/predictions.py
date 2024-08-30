from fasttext.FastText import _FastText

def parse_fasttext_label(label:str) -> tuple[str, str]:
    """
    Parse a FastText label into a code and description.

    Parameters
    ----------
    label : str
        The label to parse.

    Returns
    -------
    tuple[str, str]
        A tuple containing the code and description.

    """
    
    code:str = label.split('|')[0]
    description:str = label.split('|')[1].replace('___', ' ')

    return code, description

def print_predictions_for_products(products:list[str], model:_FastText, k:int=3, as_percentage:bool=True):
    """
    Print the predictions for the given products using the given FastText model.

    Parameters
    ----------
    products : list[str]
        The list of products to predict.
    model : _FastText
        The FastText model to use for prediction.
    k : int, optional
        The number of predictions to show, by default 3.
    as_percentage : bool, optional
        Whether to show the similarity as a percentage, by default True.

    """
    
    for product in products:
        labels, probabilities = model.predict(text=product, k=k)
        
        print(f"Producto: {product}")
        
        for i in range(k):
            label_name = labels[i].replace("__label__", "")
            label: tuple[str, str] = parse_fasttext_label(label=label_name)
            code: str = label[0]
            description: str = label[1]
            similarity = probabilities[i]
            
            if as_percentage:
                similarity_str:str = f"{similarity * 100:.2f}%"
            else:
                similarity_str:str = f"{similarity:.2f}"
                
            print(f"  {i + 1}: {code} - {description} (Similitud: {similarity_str})")
        
        print()
        
def print_nearest_neighbors(products:list[str], model:_FastText, k:int=3, as_percentage:bool=True):
    """
    Print the nearest neighbors of the given products using the given FastText model.

    Parameters
    ----------
    products : list[str]
        The list of products to find nearest neighbors for.
    model : _FastText
        The FastText model to use for finding nearest neighbors.
    k : int, optional
        The number of nearest neighbors to show, by default 3.
    as_percentage : bool, optional
        Whether to show the similarity as a percentage, by default True.

    """
    
    for product in products:
        neighbors = model.get_nearest_neighbors(word=product, k=k)
        print(f"Producto: {product}")
        
        for similarity, neighbor in neighbors:
            if as_percentage:
                similarity_str:str = f"{similarity * 100:.2f}%"
            else:
                similarity_str:str = f"{similarity:.2f}"
                
            print(f"  - {neighbor:20} (Similitud: {similarity_str})")
            
        print()