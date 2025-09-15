import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from nmf_tools.matrix_reordering.components_reordering import order_components_by_template



def get_component_data(W, W_ref=None, component_data_ref=None):
    """
    Reorder the components of W to match the components of W_ref.
    Return the component_data of W_ref reordered to match W.
    """
    if W_ref is not None:
        W_ref = np.load(W_ref).T
        component_data_ref = pd.read_table(component_data_ref)
        assert W_ref.shape[1] == W.shape[1], "W and W_ref must have the same number of samples"
        assert component_data_ref.shape[0] == W_ref.shape[0], "component_data_ref and W_ref must have the same number of components"

        if W.shape[0] > W_ref.shape[0]:
            print("Warning: W has more components than W_ref. Ignoring W_ref")
            W_ref = None

    if W_ref is None:
        component_data = pd.DataFrame({
            'index': np.arange(W.shape[0]),
            'color': define_colors(W.shape[0]),
    
        })
        component_data['name'] = "Component " + component_data['index'].astype(str)
        component_data['short_name'] = "C" + component_data['index'].astype(str)

    else:
        reorder = order_components_by_template(W, W_ref).row_order
        component_data = component_data_ref.reset_index(
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



def define_colors(n_components):
    # Function to define colors for components. Borrowed from Meuleman et al. 2020
    # Based on 2020 component colors
    comp_colors = ["#ffe500",
                    "#fe8102",
                    "#ff0000",
                    "#07af00",
                    "#4c7d14",
                    "#414613",
                    "#05c1d9",
                    "#0467fd",
                    "#009588",
                    "#bb2dd4",
                    "#7a00ff",
                    "#4a6876",
                    "#08245b",
                    "#b9461d",
                    "#692108",
                    "#c3c3c3",
                    "#6630a6",
                    "#ffc26a",
                    "#fc197e",
                    "#759cd5",
                    "#a6da57",
                    "#343331",
                    "#d04299",
                    "#a1efff",
                    "#ffadf1",
                    "#fef3bb",
                    "#61567b",
                    "#ffc9c6",
                    "#d0d2f8"]
    neworder = np.array([16,10,7,11,2,12,1,8,4,15,14,5,9,6,3,13]).astype(int) - 1
    
    component_colors = list(np.array(comp_colors)[neworder]) + comp_colors[16:]

    maxassigned = len(component_colors)
    
    if (n_components > maxassigned):
        # somewhat defunct but whatever. Adds extra "random" colors if you use more than 16 
        colornames = np.sort(list(mcolors.CSS4_COLORS.values()))
        colornames = list(set(colornames) - set(comp_colors))
        count = maxassigned
        np.random.seed(100)
        while (count < n_components):
            new_color = colornames[np.random.randint(len(colornames))]
            trialcount = 0
            while ((new_color in component_colors) and (trialcount < 100)):
                trialcount += 1
            #print('new color', count, new_color)
            component_colors.append(new_color)
            count += 1
    return component_colors[:n_components]
