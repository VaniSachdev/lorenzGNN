from utils.lorenz import lorenzDatasetWrapper

import jraph
import jax
import jax.numpy as jnp
import networkx as nx

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
import logging
import pdb


def get_lorenz_graph_tuples(predict_from,
                            n_samples,
                            input_steps,
                            output_delay,
                            output_steps,
                            timestep_duration,
                            sample_buffer,
                            time_resolution,
                            init_buffer_samples,
                            return_buffer,
                            train_pct,
                            val_pct,
                            test_pct,
                            K=36,
                            F=8,
                            c=10,
                            b=10,
                            h=1,
                            coupled=True,
                            seed=42,
                            override=False):
    """ Generated data using Lorenz96 and splits data into train/val/test. 

        Args: 
            predict_from (str): prediction paradigm. Options are "X1X2_window", 
                in which the target X1 and X2 states are predicted from the 
                input X1 and X2 states; and "X2", in which the target X1 state 
                is predicted from the input X2 state.
            n_samples (int): number of samples (windows) to generate data for.
            input_steps (int): number of timesteps in each input window.
            output_delay (int): number of timesteps strictly between the end of 
                the input window and the start of the output window.
            output_steps (int): number of timesteps in each output window.
            timestep_duration (int): the sampling rate for data points from the 
                raw Lorenz simulation data, i.e. the number of raw simulation 
                data points between consecutive timestep samples, i.e. the 
                slicing step size. all data points are separated by this value.
            sample_buffer (int): number of timesteps strictly between the end 
                of one full sample and the start of the next sample.
            time_resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            init_buffer_samples (int): number of full samples (includes input 
                and output windows) to generate before the first training 
                sample to allow for the system to settle. can be saved to use 
                or ignore during normalization.      
            return_buffer (bool): whether or not to save the buffer samples in 
                a class attribute. if saved, they will contribute to the 
                normalization step. useful to save only if generating a tiny 
                training set and need more data points for normalization; 
                otherwise, recomment discarding. 
            train_pct (float): percentage of samples to use for training.
            val_pct (float): percentage of samples to use for validation.
            test_pct (float): percentage of samples to use for testing.
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            coupled (bool): whether to use the coupled 2-layer Lorenz96 model 
                or original 1-layer Lorenz96 model
            seed (int): for reproducibility 
            override (bool): whether or not to regenerate data that was 
                already generated previously

        Output:
            returns a dict with the keys "train"/"val"/"test", each corresponding to a list. Each element of the list contains a data sample, consisting of another dictionary containing "input_graphs" and "target_graphs" as keys; the values are lists of jraph.GraphsTuple objects, corresponding to the input graphs and target graphs datapoints in the sample. 
    """
    logging.debug('Generating graph tuples')
    # i'm just going to pull the data out of a lorenz spektral dataset
    # this is computationally inefficient but convenient code-wise so I don't
    # have to rewrite all the normalization functions and stuff
    dataset = lorenzDatasetWrapper(predict_from=predict_from,
                                   n_samples=n_samples,
                                   input_steps=input_steps,
                                   output_delay=output_delay,
                                   output_steps=output_steps,
                                   timestep_duration=timestep_duration,
                                   sample_buffer=sample_buffer,
                                   time_resolution=time_resolution,
                                   init_buffer_samples=init_buffer_samples,
                                   return_buffer=return_buffer,
                                   train_pct=train_pct,
                                   val_pct=val_pct,
                                   test_pct=test_pct,
                                   K=K,
                                   F=F,
                                   c=c,
                                   b=b,
                                   h=h,
                                   coupled=coupled,
                                   seed=seed,
                                   override=override)
    # note that we don't care about the preprocessing or simple_adj arguments 
    # because we're not using the adjacency matrix generated in the 
    # lorenzDatasetWrapper class
    dataset.normalize()

    graph_tuple_dict = {'train': {'inputs': [], 'targets': []}, 
                             'val': {'inputs': [], 'targets': []}, 
                             'test': {'inputs': [], 'targets': []}}

    for g in dataset.train:
        # g.x has shape 36 x 2
        input_graph_tuple = timestep_to_graphstuple(g.x, K=K)
        output_graph_tuple = timestep_to_graphstuple(g.y , K=K)
        graph_tuple_dict['train']['inputs'].append(input_graph_tuple)
        graph_tuple_dict['train']['targets'].append(output_graph_tuple)

    if dataset.val is not None:
        # TODO pick up here
        for g in dataset.val:
            # g.x has shape 36 x 2
            graph_tuple = timestep_to_graphstuple(g.x, K=K)
            graph_tuple_dict['val'].append(graph_tuple)

    if dataset.test is not None:
        for g in dataset.test:
            # g.x has shape 36 x 2
            graph_tuple = timestep_to_graphstuple(g.x, K=K)
            graph_tuple_dict['test'].append(graph_tuple)

    return graph_tuple_dict


def timestep_to_graphstuple(data, K):
    """ Converts an array of state values at a single timestep to a GraphsTuple 
        object.
    
        Args:
                data: array of shape (K, num_fts)
                K (int): number of nodes in the Lorenz system
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
        globals=jnp.array([[1.]]),  # placeholder global features for now (was an empty array and None both causing errors down the line?)
        # globals=jnp.array([]),  # no global features for now
        # globals=None,  # no global features for now
        nodes=jnp.array(
            data),  # node features = state values. shape of (K, 2)
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


# def get_data_windows(graph_tuple_list, n_rollout_steps, timestep_duration: int):
#     """ Get inputs and targets from a graph_tuple containing time series data
    
#         Args: 
#             graph_tuple_list: dist of lists of GraphsTuple objects
#             n_rollout_steps (int): number of steps for rollout output

#         Returns:
#             inputs, 1D list of GraphTuples, with length 
#                 (datapoints - n_rollout_steps * timestep_duration)
#             targets, 2D list of GraphTuples, with shape 
#                 (datapoints - n_rollout_steps * timestep_duration, n_rollout_steps)
#     """
#     inputs = []
#     targets = []
#     # TODO: maybe we should convert these to a pd dataframe?
#     # this also seems quite space-inefficient

#     orig_timesteps = len(graph_tuple_list)
#     n_timesteps = orig_timesteps - n_rollout_steps * timestep_duration

#     # print(orig_timesteps, n_timesteps)
#     for i in range(n_timesteps):
#         input_graph = graph_tuple_list[i]
#         target_graphs = graph_tuple_list[i + timestep_duration:i +
#                                          (1 + n_rollout_steps) *
#                                          timestep_duration:timestep_duration]
#         # print(type(input_graph))
#         # print(type(target_graphs))
#         inputs.append(input_graph)
#         targets.append(target_graphs)

#     # return np.concatenate(inputs, axis=0, dtype=object), np.concatenate(targets, axis=0, dtype=object)
#     # return np.vstack(inputs), np.vstack(targets)
#     # print('len(inputs), len(targets)', len(inputs), len(targets))
#     return inputs, targets


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    """ Converts a jraph GraphsTuple object to a networkx graph object."""
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