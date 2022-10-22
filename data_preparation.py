import numpy as np

def getLorenzGraph(X, verbose=False):
    """ Create a graph representation of the Lorenz data at a single instance in
        time. 

        The graph contains: 
        * edges: an adjacency list, i.e. a 2d array of size (2, num_edges). The
          first row contains the indices of the source node in each edge, and
          the second row contains the indices of the target node in each edge. 
        * node_features: 2d array of size (num_nodes, num_features). Each row
          contains an array of features for the node. 
        * edge_weights: 1d array of size (num_edges, ). Edge weights shouldn't
          be relevant to our problem so we set all to 1. 

        Args: 
            X (1D float array, size 2*K): array of current X and Y state values

        Returns: 3-tuple containing graph information
    """
    K = len(X) // 2

    edges = np.array([
        [i for i in range(K)]*2, # source nodes
        [i+1 for i in range(K-1)] + [0] + [K-1] + [i-1 for i in range(1, K)] 
            # ^ target nodes (i.e. left and right neighbors)
    ])
    edge_weights = np.ones(shape=edges.shape[1])
    node_features = np.reshape(X, (2, K)).T

    graph_info = (node_features, edges, edge_weights)

    if verbose:
        print('edges\n', edges)
        print('X\n', X)
        print('node_features\n', node_features)

        print("Edges shape:", edges.shape)
        print("Nodes shape:", node_features.shape)

    return graph_info