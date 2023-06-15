import numpy as np

import matplotlib.pyplot as plt
from lorenz import lorenzDatasetWrapper
from plotters import plot_data

import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk

import functools
import optax
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable


def lorenz_graph_tuple_list(n_samples=2_000):
    # i'm just going to pull the data out of a lorenz spektral dataset
    # this is computationally inefficient but convenient code-wise so I don't
    # have to rewrite all the normalization functions and stuff

    # only uncomment each line if testing a non-default parameter
    K = 36
    dataset = lorenzDatasetWrapper(
        predict_from="X1X2_window",
        n_samples=n_samples,
        input_steps=1,
        output_delay=0,
        output_steps=0,
        min_buffer=0,
        # rand_buffer=False,
        K=K,
        # F=8,
        # c=10,
        # b=10,
        # h=1,
        # coupled=True,
        # time_resolution=DEFAULT_TIME_RESOLUTION, #(100)
        # seed=42,
        init_buffer_steps=100,
        return_buffer=False,
        train_pct=0.7,
        val_pct=0.3,
        test_pct=0,
        override=False)
    dataset.normalize()

    # iter over time steps in lorenz data

    graph_tuple_lists = {'train': [], 'val': []}
    # construct a new data dict for each time step
    for g in dataset.train:
        # g.x has shape 36 x 2
        graph_tuple = timestep_to_graphstuple(g.x, K=K)
        graph_tuple_lists['train'].append(graph_tuple)
    for g in dataset.val:
        # g.x has shape 36 x 2
        graph_tuple = timestep_to_graphstuple(g.x, K=K)
        graph_tuple_lists['val'].append(graph_tuple)

    # convert all data dicts into a single GraphTuple
    # graph_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)

    # return graph_tuple
    return graph_tuple_lists


def timestep_to_graphstuple(data, K):
    """ Args:
            data: array of shape (K, num_fts)
            K (int): number of nodes in the Lorenz system
            ft_type (str): either "global" or "nodes", i.e. whether the 
                node-wise features of the Lorenz system should be represented using node features or global features in the GN approach. 
    """
    # define edges
    receivers = []
    senders = []
    edge_fts = []

    for i in range(K):
        senders += [i] * 5
        receivers += [i, (i + 1) % K, (i + 2) % K, (i - 1) % K, (i - 2) % K]

        # edge features = length + direction of edge
        edge_fts += [
            [0],  # self edge
            [1],  # receiver is 1 node to the right
            [2],  # receiver is 2 nodes to the right
            [-1],  # receiver is 1 node to the left
            [-2]  # receiver is 2 nodes to the left
        ]

    return jraph.GraphsTuple(
        globals=jnp.array([[1.]]),  # placeholder global features for now
        # globals=jnp.array([]),  # no global features for now
        # globals=None,  # no global features for now
        nodes=jnp.array(
            data),  # node features = atmospheric measurements. shape of (K, 2)
        edges=jnp.array(edge_fts, dtype=float),
        receivers=jnp.array(receivers),
        senders=jnp.array(senders),
        n_node=jnp.array([K]),
        n_edge=jnp.array([K * 5]))


def print_graph_fts(graph: jraph.GraphsTuple):
    print(f'Number of nodes: {graph.n_node[0]}')
    print(f'Number of edges: {graph.n_edge[0]}')
    print(f'Node features shape: {graph.nodes.shape}')
    print(f'Edge features shape: {graph.edges.shape}')
    print(f'Global features shape: {graph.globals.shape}')


def get_data_windows(graph_tuple_list, n_rollout_steps, timestep_duration: int):
    """ Get inputs and targets from a graph_tuple containing time series data
    
        Args: 
            graph_tuple_list: dist of lists of GraphsTuple objects
            n_rollout_steps (int): number of steps for rollout output

        Returns:
            inputs, 1D list of length (timesteps - 2 - n_rollout_steps)
            targets, 2D list of size (timesteps - 2 - n_rollout_steps, n_rollout_steps)
    """
    inputs = []
    targets = []
    # TODO: maybe we should convert these to a pd dataframe?
    # this also seems quite space-inefficient

    orig_timesteps = len(graph_tuple_list)
    n_timesteps = orig_timesteps - n_rollout_steps * timestep_duration

    print(orig_timesteps, n_timesteps)
    for i in range(n_timesteps):
        input_graph = graph_tuple_list[i]
        target_graphs = graph_tuple_list[i + timestep_duration:i +
                                         (1 + n_rollout_steps) *
                                         timestep_duration:timestep_duration]
        # print(type(input_graph))
        # print(type(target_graphs))
        inputs.append(input_graph)
        targets.append(target_graphs)

    # return np.concatenate(inputs, axis=0, dtype=object), np.concatenate(targets, axis=0, dtype=object)
    # return np.vstack(inputs), np.vstack(targets)
    # print('len(inputs), len(targets)', len(inputs), len(targets))
    return inputs, targets


def data_list_to_dict(graph_tuple_list: Iterable[jraph.GraphsTuple],
                      n_rollout_steps: int, timestep_duration: int):
    inputs_list, targets_list = get_data_windows(graph_tuple_list,
                                                 n_rollout_steps,
                                                 timestep_duration)
    # print(len(inputs_list))
    # print(len(targets_list))
    # print(len(targets_list[0]))
    # print(type(targets_list[0]))
    # print(len(targets_list[0][0]))
    # print(type(targets_list[0][0]))
    data_dict_list = [
        {
            'input_graph': inputs_list[i],  # input is a single graph
            'target': targets_list[i]
        }  # target is single graph for now while we test
        for i in range(len(inputs_list))
    ]

    return data_dict_list


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]),
                              int(receivers[e]),
                              edge_feature=edges[e])
    return nx_graph