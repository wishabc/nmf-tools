import numpy as np
import pandas as pd

from nmf_tools.matrix.reorder import reorder_components

W_old_path_default = '/net/seq/data2/projects/sabramov/SuperIndex/dnase-peak-calls/embeddings/NMF/p_weights.0.1pr_index/output/nmf/dhs_point1pr.28/dhs_point1pr.28.W.npy'
component_data_old_path_default = '/home/sabramov/temp_component_metadata_dhs_point1pr_28.tsv'


def get_component_data(W, W_old_path=W_old_path_default, component_data_old_path=component_data_old_path_default):
    '''
    Reorder the components of W to match the components of W_old.
    Return the component_data of W_old reordered to match W.
    '''
    W_old = np.load(W_old_path)
    component_data_old = pd.read_table(component_data_old_path)

    reorder = reorder_components(W, W_old)

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