from .components_reordering import order_components
from .records_reordering import order_records
from .matrix_reordering import MatrixReordering

def order_matrix(matrix,
                 order_components_by='primary',
                 order_components_kwargs=None,
                 order_records_by='primary',
                 order_records_kwargs=None,
                 normalize_for_plotting=True):
    
    if order_components_kwargs is None:
        order_components_kwargs = {}
    if order_records_kwargs is None:
        order_records_kwargs = {}
    
    components_order = order_components(matrix,
                            by=order_components_by,
                            **order_components_kwargs)

    records_order = order_records(matrix,
                            by=order_records_by,
                            **order_records_kwargs)
    
    if normalize_for_plotting:
        matrix = matrix / matrix.sum(axis=0, keepdims=True)
   
    sorted_matrix = components_order(matrix)
    tops = sorted_matrix.cumsum(axis=0)
    bottoms = tops - sorted_matrix

    inverse_component_order = components_order.inv
    tops = records_order(inverse_component_order(tops))
    bottoms = records_order(inverse_component_order(bottoms))
  
    return tops, bottoms, components_order, records_order
