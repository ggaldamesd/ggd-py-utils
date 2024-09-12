from pandas import DataFrame
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
    show_in_browser:bool = False,
    title:str = None,
    zoom_factor:float = 0.5
) -> None:
    """
    Visualizes embeddings from a DataFrame using PCA for dimensionality reduction 
    and plots them interactively with Plotly. It also performs a search query by 
    calculating the cosine similarity between the search embedding and the 
    embeddings in the DataFrame.

    Displays an interactive 2D or 3D plot showing the top `k` similar 
    embeddings to the search term, colored by similarity, along with the 
    search embedding.
    
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
        
    show_in_browser : str, optional (default=False)
        If True, the plot will be displayed in the browser.
        
    zoom_factor : float, optional (default=0.5)
        The zoom factor to use when displaying the plot.

    Returns:
    --------
    None
    """
    from ggd_py_utils.tracing.metrics import time_block

    with time_block(block_name="Cleaning and Getting Embeddings."):
        from ggd_py_utils.machine_learning.data.cleaning import clean_text

        clean_search: str = clean_text(text=search)
        search_embedding = model.get_sentence_vector(text=clean_search).tolist()
    
    with time_block(block_name="Calculating Cosine Similarities."):
        from scipy.spatial.distance import cosine

        df[similarity_field_name] = df[embedding_field_name].apply(lambda x: 1 - cosine(search_embedding, x))
        min_sim, max_sim = df[similarity_field_name].min(), df[similarity_field_name].max()
        df[similarity_field_name] = (df[similarity_field_name] - min_sim) / (max_sim - min_sim)

    with time_block(block_name="Filtering Similarities by Threshold."):
        df_filtered: DataFrame = df[df[similarity_field_name] >= threshold].sort_values(similarity_field_name, ascending=False)
        
        if df_filtered.empty:
            print(f"No results found for search with threshold {threshold*100:.2f}: {search}")
            return
        
        df_top: DataFrame = df_filtered.head(k)
    
    embeddings:list = df_top[embedding_field_name].values.tolist()
    n_components:int = 3 if plot_in_3d else 2
    
    with time_block(block_name="Reducing Embeddings Dimensions."):
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=42)
        
        from numpy import ndarray

        reduced_embeddings: ndarray = reducer.fit_transform(embeddings)
    
        from numpy import array

        search_embedding_reshaped: ndarray = array(search_embedding).reshape(1, -1)
        reduced_search_embedding: ndarray = reducer.transform(search_embedding_reshaped)
    
    with time_block(block_name="Plotting."):
        from plotly.graph_objects import Figure

        fig = Figure()

        color_maps:list = [
            "aggrnyl", 
            "agsunset", 
            "algae", 
            "amp", 
            "armyrose", 
            "balance", 
            "blackbody", 
            "bluered",
            "blues",
            "blugrn",
            "bluyl",
            "brbg",
            "brwnyl",
            "bugn", 
            "bupu", 
            "burg", 
            "burgyl", 
            "cividis", 
            "curl", 
            "darkmint", 
            "deep", 
            "delta", 
            "dense", 
            "earth", 
            "edge", 
            "electric", 
            "emrld", 
            "fall", 
            "geyser", 
            "gnbu", 
            "gray", 
            "greens", 
            "greys", 
            "haline", 
            "hot", 
            "hsv", 
            "ice", 
            "icefire", 
            "inferno", 
            "jet", 
            "magenta", 
            "magma", 
            "matter", 
            "mint", 
            "mrybm", 
            "mygbm", 
            "oranges", 
            "orrd", 
            "oryel", 
            "oxy",
            "peach", 
            "phase", 
            "picnic", 
            "pinkyl", 
            "piyg", 
            "plasma", 
            "plotly3", 
            "portland", 
            "prgn", 
            "pubu", 
            "pubugn", 
            "puor", 
            "purd", 
            "purp", 
            "purples", 
            "purpor", 
            "rainbow", 
            "rdbu", 
            "rdgy", 
            "rdpu", 
            "rdylbu", 
            "rdylgn", 
            "redor", 
            "reds", 
            "solar", 
            "spectral", 
            "speed", 
            "sunset", 
            "sunsetdark", 
            "teal", 
            "tealgrn", 
            "tealrose", 
            "tempo", 
            "temps", 
            "thermal", 
            "tropic", 
            "turbid", 
            "turbo", 
            "twilight", 
            "viridis", 
            "ylgn", 
            "ylgnbu", 
            "ylorbr", 
            "ylorrd",
            "aggrnyl_r", 
            "agsunset_r", 
            "algae_r", 
            "amp_r", 
            "armyrose_r", 
            "balance_r", 
            "blackbody_r", 
            "bluered_r",
            "blues_r",
            "blugrn_r",
            "bluyl_r",
            "brbg_r",
            "brwnyl_r",
            "bugn_r", 
            "bupu_r", 
            "burg_r", 
            "burgyl_r", 
            "cividis_r", 
            "curl_r", 
            "darkmint_r", 
            "deep_r", 
            "delta_r", 
            "dense_r", 
            "earth_r", 
            "edge_r", 
            "electric_r", 
            "emrld_r", 
            "fall_r", 
            "geyser_r", 
            "gnbu_r", 
            "gray_r", 
            "greens_r", 
            "greys_r", 
            "haline_r", 
            "hot_r", 
            "hsv_r", 
            "ice_r", 
            "icefire_r", 
            "inferno_r", 
            "jet_r", 
            "magenta_r", 
            "magma_r", 
            "matter_r", 
            "mint_r", 
            "mrybm_r", 
            "mygbm_r", 
            "oranges_r", 
            "orrd_r", 
            "oryel_r", 
            "oxy_r",
            "peach_r", 
            "phase_r", 
            "picnic_r", 
            "pinkyl_r", 
            "piyg_r", 
            "plasma_r", 
            "plotly3_r", 
            "portland_r", 
            "prgn_r", 
            "pubu_r", 
            "pubugn_r", 
            "puor_r", 
            "purd_r", 
            "purp_r", 
            "purples_r", 
            "purpor_r", 
            "rainbow_r", 
            "rdbu_r", 
            "rdgy_r", 
            "rdpu_r", 
            "rdylbu_r", 
            "rdylgn_r", 
            "redor_r", 
            "reds_r", 
            "solar_r", 
            "spectral_r", 
            "speed_r", 
            "sunset_r", 
            "sunsetdark_r", 
            "teal_r", 
            "tealgrn_r", 
            "tealrose_r", 
            "tempo_r", 
            "temps_r", 
            "thermal_r", 
            "tropic_r", 
            "turbid_r", 
            "turbo_r", 
            "twilight_r", 
            "viridis_r", 
            "ylgn_r", 
            "ylgnbu_r", 
            "ylorbr_r", 
            "ylorrd_r"
        ]
        
        _color_map = color_map
        
        if color_map in color_maps:
            _color_map = color_map
        elif color_map == "random":
            from random import choice

            _color_map:str = choice(color_maps)
            print(_color_map)
        else:
            _color_map = "plasma_r"
        
        if metadata_fields:
            from pandas import Series

            hover_text: Series[str] = df_top.apply(
                lambda row: '<br>'.join([f"{field}: {row[field]}" for field in metadata_fields]) + 
                        f"<br>Similitud: {row[similarity_field_name]*100:.2f}%", axis=1
            ) 
        else:
            hover_text: Series[str] = df_top.apply(
                lambda row: f"Similitud: {row[similarity_field_name]*100:.2f}%", axis=1
            )
        
        similarity: ndarray = df_top[similarity_field_name].values
 
        best_nodes: DataFrame = df_top.nlargest(1, similarity_field_name)
        best_node_index = best_nodes.index[0]
        best_node_embedding:ndarray = reduced_embeddings[df_top.index.get_loc(best_node_index)]

        x_best = best_node_embedding[0] + zoom_factor
        y_best = best_node_embedding[1] + zoom_factor
        z_best = best_node_embedding[2] + zoom_factor if plot_in_3d else None

        if plot_in_3d:
            from plotly.graph_objects import Scatter3d

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
                        tickformat=".0%"
                    ),
                    showscale=True
                ),
                text=hover_text,
                textposition='top center',
                hoverinfo='text'
            )

            highlighted_scatter = Scatter3d(
                x=[best_node_embedding[0]],
                y=[best_node_embedding[1]],
                z=[best_node_embedding[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    opacity=1,
                    symbol='x'
                ),
                text=hover_text,
                textposition='top center',
                hoverinfo='text'
            )
        else:
            from plotly.graph_objects import Scatter

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
                        tickformat='.0%'
                    ),
                    showscale=True
                ),
                text=hover_text,
                textposition='top center',
                hoverinfo='text'
            )

            highlighted_scatter = Scatter(
                x=[best_node_embedding[0]],
                y=[best_node_embedding[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    opacity=1,
                    symbol='x'
                ),
                text=hover_text,
                textposition='top center',
                hoverinfo='text'
            )
            
        scene_camera = dict(
            eye=dict(
                x=x_best,
                y=y_best,
                z=z_best
            ),
            center=dict(
                x=best_node_embedding[0],
                y=best_node_embedding[1],
                z=best_node_embedding[2] if plot_in_3d else None
            ),
            projection=dict(type='perspective')
        )

        fig.add_trace(scatter)

        scene_axis = dict(
            backgroundcolor='white',
            title="", 
            showticklabels=False, 
            showgrid=False,
            showline=False, 
            zeroline=False,
            tickvals=[],
            ticktext=[],
            ticks=""
        )

        fig.update_layout(
            title=title,
            showlegend=False,
            width=800,
            height=600,
            scene_camera=scene_camera,
            plot_bgcolor="white",
            paper_bgcolor="white",
            scene=dict(
                xaxis=scene_axis,
                yaxis=scene_axis,
                zaxis=scene_axis if plot_in_3d else None
            )
        )

        fig.show(renderer="browser" if show_in_browser else None)
