from fasttext.FastText import _FastText

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