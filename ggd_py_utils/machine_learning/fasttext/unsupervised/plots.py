from sklearn.decomposition import PCA
from plotly.graph_objects import Figure, Scatter3d, Scatter
from scipy.spatial.distance import cosine
from ggd_py_utils.machine_learning.data.cleaning import clean_text
from pandas import DataFrame, Series
from numpy import ndarray, array
from random import choice
from fasttext.FastText import _FastText

def plot_embeddings_with_search(
    df: DataFrame, 
    model: _FastText, 
    search: str, 
    threshold: float = 0.5, 
    similarity_field_name: str = "Similarity",
    embedding_field_name: str = "Embeddings",
    metadata_fields: list[str] = [],
    k:int = 50,
    color_map:str = "random",
    plot_in_3d:bool = False,
    title:str = None
) -> None:
    """
    Visualizes embeddings from a DataFrame using PCA for dimensionality reduction 
    and plots them interactively with Plotly. It also performs a search query by 
    calculating the cosine similarity between the search embedding and the 
    embeddings in the DataFrame.

    Parameters:
    -----------
    df : DataFrame
        A pandas DataFrame containing a column of embeddings and any other metadata.
    
    model : _FastText
        A FastText model used to generate the embedding vector for the search query.
    
    search : str
        The search term to compare against the embeddings in the DataFrame.
    
    threshold : float, optional (default=0.5)
        The minimum similarity score required to include a result in the final plot.
    
    similarity_field_name : str, optional (default="Similarity")
        The name of the field where similarity scores will be stored.
    
    embedding_field_name : str, optional (default="Embeddings")
        The name of the column in the DataFrame that contains the embedding vectors.
    
    metadata_fields : list[str], optional (default=[])
        A list of field names from the DataFrame to include as hover text in the plot.
    
    k : int, optional (default=50)
        The maximum number of top similar embeddings to display.
    
    color_map : str, optional (default="random")
        The color map used for visualizing similarity in the plot. Can be one of 
        the predefined maps or "random" to choose a random one.
    
    plot_in_3d : bool, optional (default=False)
        If True, the plot will be in 3D; otherwise, it will be 2D.

    title : str, optional (default=None)
        The plot title.

    Returns:
    --------
    None
        Displays an interactive 2D or 3D plot showing the top `k` similar 
        embeddings to the search term, colored by similarity, along with the 
        search embedding.
    """
    clean_search: str = clean_text(text=search)
    search_embedding = model.get_sentence_vector(text=clean_search).tolist()
    
    df[similarity_field_name] = df[embedding_field_name].apply(lambda x: 1 - cosine(search_embedding, x))
    min_sim, max_sim = df[similarity_field_name].min(), df[similarity_field_name].max()
    df[similarity_field_name] = (df[similarity_field_name] - min_sim) / (max_sim - min_sim)

    df_filtered: DataFrame = df[df[similarity_field_name] >= threshold].sort_values(similarity_field_name, ascending=False)
    
    if df_filtered.empty:
        print(f"No results found for search with threshold {threshold*100:.2f}: {search}")
        return
    
    df_top: DataFrame = df_filtered.head(k)
    
    embeddings:list = df_top[embedding_field_name].values.tolist()
    n_components:int = 3 if plot_in_3d else 2
    
    reducer = PCA(n_components=n_components, random_state=42)
    
    reduced_embeddings: ndarray = reducer.fit_transform(embeddings)
    
    search_embedding_reshaped: ndarray = array(search_embedding).reshape(1, -1)
    reduced_search_embedding: ndarray = reducer.transform(search_embedding_reshaped)
    
    fig = Figure()

    color_maps:list = [
        "Plasma_r",
        "Viridis_r",
        "Inferno_r",
        "Turbo_r",
        "Jet_r",
        "RdBu_r",
        "YlOrRd_r",
        "Hot_r"
    ]
    
    _color_map = color_map
    
    if color_map in color_maps:
        _color_map = color_map
    elif color_map == "random":
        _color_map:str = choice(color_maps)
    else:
        _color_map = "YlOrRd"
    
    if metadata_fields:
        hover_text: Series[str] = df_top.apply(
            lambda row: ', '.join([f"{field}: {row[field]}" for field in metadata_fields]) + 
                    f", Similitud: {row[similarity_field_name]*100:.2f}%", axis=1
        ) 
    else:
        hover_text: Series[str] = df_top.apply(
            lambda row: f"Similitud: {row[similarity_field_name]*100:.2f}%", axis=1
        )
    
    similarity: ndarray = df_top[similarity_field_name].values
    
    if plot_in_3d:
        scatter = Scatter3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=10 * similarity,
                opacity=1,
                color=similarity,
                colorscale=_color_map,
                colorbar=dict(
                    orientation='h',
                    title="Similitud",
                ),
                showscale=True
            ),
            text=hover_text,
            textposition='top center'
        )
        
        search_scatter = Scatter3d(
            x=[reduced_search_embedding[0][0]],
            y=[reduced_search_embedding[0][1]],
            z=[reduced_search_embedding[0][2]],
            mode='markers+text',
            marker=dict(
                size=10, 
                opacity=1,
                color=[max(similarity)],
            ),
            text=[search],
            textposition='top center'
        )
    else:
        scatter = Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=10 * similarity,
                opacity=1,
                color=similarity,
                colorscale=_color_map,
                colorbar=dict(
                    orientation='h',
                    title="Similitud",
                ),
                showscale=True
            ),
            text=hover_text,
            textposition='top center'
        )
        
        search_scatter = Scatter(
            x=[reduced_search_embedding[0][0]],
            y=[reduced_search_embedding[0][1]],
            mode='markers+text',
            marker=dict(
                size=10, 
                opacity=1,
                color=[max(similarity)],
            ),
            text=[search],
            textposition='top center'
        )

    fig.add_trace(scatter)
    fig.add_trace(search_scatter)

    fig.update_layout(
        title=title,
        showlegend=False,
        width=800,
        height=600
    )

    fig.show()
