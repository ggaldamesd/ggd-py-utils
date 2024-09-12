from pandas import DataFrame
from fasttext.FastText import _FastText

def plot_embeddings_with_search(
    df: DataFrame, 
    model: _FastText, 
    search_text: str, 
    threshold: float = 0.5, 
    similarity_field_name: str = "Similarity",
    embedding_field_name: str = "Embeddings",
    metadata_fields: list[str] = [],
    k:int = 50,
    color_map:str = "plasma_r",
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

    Args:
        df (DataFrame): A pandas DataFrame containing a column of embeddings and any other metadata.
        model (_FastText): A FastText model used to generate the embedding vector for the search query.
        search_text (str): The search term to compare against the embeddings in the DataFrame.
        threshold (float, optional): The minimum similarity score required to include a result in the final plot. Defaults to 0.5.
        similarity_field_name (str, optional): The name of the field where similarity scores will be stored. Defaults to "Similarity".
        embedding_field_name (str, optional): The name of the column in the DataFrame that contains the embedding vectors. Defaults to "Embeddings".
        metadata_fields (list of str, optional): A list of field names from the DataFrame to include as hover text in the plot. Defaults to an empty list.
        k (int, optional): The maximum number of top similar embeddings to display. Defaults to 50.
        color_map (str, optional): The color map used for visualizing similarity in the plot. Can be one of the predefined maps or "random" to choose a random one. Defaults to "plasma_r".
        plot_in_3d (bool, optional): If True, the plot will be in 3D; otherwise, it will be 2D. Defaults to False.
        title (str, optional): The plot title. Defaults to None.
        show_in_browser (bool, optional): If True, the plot will be displayed in the browser. Defaults to False.
        zoom_factor (float, optional): The zoom factor to use when displaying the plot. Defaults to 0.5.

    Returns:
        None

    Example:
        >>> from ggd_py_utils.machine_learning.unsupervised.plots import plot_embeddings_with_search
        >>> search_text = "clavo teja"
        >>> plot_embeddings_with_search(
        ...     df=df,
        ...     model=model,
        ...     threshold=0.01,
        ...     search_text=search_text,
        ...     metadata_fields=["NombreClase", "NombreProducto"],
        ...     k=10000,
        ...     color_map="plasma_r",
        ...     plot_in_3d=True,
        ...     show_in_browser=False,
        ...     title="Espacio Vectorial de Productos y Categorías UNSPSC",
        ...     zoom_factor=0.99
        ... )
        >>> plot_embeddings_with_search(
        ...     df=df,
        ...     model=model,
        ...     threshold=0.5,
        ...     search_text=search_text,
        ...     metadata_fields=["NombreClase", "NombreProducto"],
        ...     k=1000,
        ...     color_map="plasma_r",
        ...     plot_in_3d=False,
        ...     show_in_browser=False,
        ...     title="Espacio Vectorial de Productos y Categorías UNSPSC",
        ...     zoom_factor=0.99
        ... )
    """
    from ggd_py_utils.tracing.metrics import time_block

    with time_block(block_name="Cleaning and Getting Embeddings."):
        from ggd_py_utils.machine_learning.data.cleaning import clean_text

        clean_search: str = clean_text(text=search_text)
        search_embedding = model.get_sentence_vector(text=clean_search).tolist()
    
    with time_block(block_name="Calculating Cosine Similarities."):
        from scipy.spatial.distance import cosine

        df[similarity_field_name] = df[embedding_field_name].apply(lambda x: 1 - cosine(search_embedding, x))
        min_sim, max_sim = df[similarity_field_name].min(), df[similarity_field_name].max()
        df[similarity_field_name] = (df[similarity_field_name] - min_sim) / (max_sim - min_sim)

    with time_block(block_name="Filtering Similarities by Threshold."):
        df_filtered: DataFrame = df[df[similarity_field_name] >= threshold].sort_values(similarity_field_name, ascending=False)
        
        if df_filtered.empty:
            print(f"No results found for search with threshold {threshold*100:.2f}: {search_text}")
            return
        
        df_top: DataFrame = df_filtered.head(k)
    
    embeddings:list = df_top[embedding_field_name].values.tolist()
    n_components:int = 3 if plot_in_3d else 2
    
    with time_block(block_name="Reducing Embeddings Dimensions."):
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=42)
        
        from numpy import ndarray

        reduced_embeddings: ndarray = reducer.fit_transform(embeddings)
    
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

        best_node_pos = df_top.index.get_loc(best_node_index)
        
        from numpy import delete
        reduced_embeddings = delete(reduced_embeddings, best_node_pos, axis=0)

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
                    size=5,
                    color='green',
                    opacity=1,
                    symbol='diamond'
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
                    size=10,
                    color='green',
                    opacity=1,
                    symbol='diamond'
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
        fig.add_trace(highlighted_scatter)

        if plot_in_3d:
            scene_axis_3d = dict(
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
        
        if not plot_in_3d:
            x_scene_axis_2d = dict(
                title="", 
                showticklabels=False, 
                showgrid=False,
                showline=False, 
                zeroline=False,
                tickvals=[],
                ticktext=[],
                ticks="",
                # range=[x_best - 0.1, x_best + 0.1]
            )
            
            y_scene_axis_2d = dict(
                title="", 
                showticklabels=False, 
                showgrid=False,
                showline=False, 
                zeroline=False,
                tickvals=[],
                ticktext=[],
                ticks="",
                # range=[y_best - 0.1, y_best + 0.1]
            )

        title = f"Espacio Vectorial para: <b>{search_text}</b>" if None else f"{title} para: <b>{search_text}</b>"

        fig.update_layout(
            title=title,
            showlegend=False,
            width=800,
            height=600,
            scene_camera=scene_camera if plot_in_3d else None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=x_scene_axis_2d if not plot_in_3d else None,
            yaxis=y_scene_axis_2d if not plot_in_3d else None,
            scene=dict(
                xaxis=scene_axis_3d,
                yaxis=scene_axis_3d,
                zaxis=scene_axis_3d 
            ) if plot_in_3d else None
        )

        fig.show(renderer="browser" if show_in_browser else None)
