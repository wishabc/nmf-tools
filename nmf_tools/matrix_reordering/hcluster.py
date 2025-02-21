import scipy.cluster.hierarchy as sch

def hierarchical_clustering(W, metric='cosine', method='average', **kwargs):
    '''
    Perform hierarchical clustering on the records of W.
    '''
    distance_matrix = sch.distance.pdist(W.T, metric=metric)
    linkage_matrix = sch.linkage(distance_matrix, method=method, **kwargs)
    return linkage_matrix