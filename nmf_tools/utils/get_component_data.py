import numpy as np
import pandas as pd

from nmf_tools.matrix_reordering.components_reordering import order_components_by_template

W_old_path_default = '/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/embeddings/NMF/meta_mar5_2025/output/nmf/hotspot3_index.mar5.filled_1pr.ontology_extended_spot_30_weights_norm.30/hotspot3_index.mar5.filled_1pr.ontology_extended_spot_30_weights_norm.30.W.npy'
component_data_old_path_default = '/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/embeddings/NMF/meta_mar5_2025/output/nmf/hotspot3_index.mar5.filled_1pr.ontology_extended_spot_30_weights_norm.30/component_data.tsv'


def get_component_data(W, W_old_path=W_old_path_default, component_data_old_path=component_data_old_path_default):
    '''
    Reorder the components of W to match the components of W_old.
    Return the component_data of W_old reordered to match W.
    '''
    W_old = np.load(W_old_path)
    component_data_old = pd.read_table(component_data_old_path)

    reorder = order_components_by_template(W, W_old).row_order

    component_data = pd.DataFrame({
        'index': component_data_old['index'],
        'color': component_data_old['color'],
        'name': component_data_old['name'],
    })

    component_data = component_data.reset_index(
        names='color_order'
    ).set_index('index').loc[
        reorder
    ].reset_index(
        drop=True
    ).reset_index(
        names='index'
    ).sort_values(
        'color_order'
    ).reset_index(
        drop=True
    ).drop(
        columns='color_order'
    )

    return component_data