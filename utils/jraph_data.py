from utils.lorenz import DATA_DIRECTORY_PATH, run_download_lorenz96_2coupled, load_lorenz96_2coupled, get_window_indices, normalize_lorenz96_2coupled

import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np 
import json 
import os 
from datetime import datetime
from functools import partial

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
import logging
import pdb

def get_lorenz_graph_tuples(n_samples,
                            input_steps,
                            output_delay,
                            output_steps,
                            timestep_duration,
                            sample_buffer,
                            time_resolution,
                            init_buffer_samples,
                            train_pct,
                            val_pct,
                            test_pct,
                            K=36,
                            F=8,
                            c=10,
                            b=10,
                            h=1,
                            seed=42,
                            normalize=False,
                            fully_connected_edges=True,
                            # data_path=None,
                            ):
    """ Generated data using Lorenz96 and splits data into train/val/test. 

        Args: 
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
            normalize (bool): whether or not to normalize the data.
            # data_path (str): optional file path. if None, will iterate over all existing simulation data to find a valid dataset with compatible parameters, or generate new simulation data if it cannot find any (using a default generated data path). if a path is given, then the simulation data will be checked to see if it exists is compatible; if it doesn't exist, it will generate new simulation data at that path; if it exists but was incompatible, an error will be raised. 

        Output:
            returns a dict with the keys "train"/"val"/"test", each corresponding to a list. Each element of the list contains a data sample, consisting of another dictionary containing "input_graphs" and "target_graphs" as keys; the values are lists of jraph.GraphsTuple objects, corresponding to the input graphs and target graphs datapoints in the sample. 
    """
    logging.debug('Generating graph tuples')
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001
    # use error term due to float errors

    # compute the number of total "raw" steps in the simulation we need
    # TODO test if this section is correct 
    total_samples = n_samples + init_buffer_samples

    all_input_window_indices, all_target_window_indices = get_window_indices(
            n_samples=total_samples, 
            timestep_duration=timestep_duration, input_steps=input_steps, output_delay=output_delay, output_steps=output_steps, sample_buffer=sample_buffer)
    # note that window indices include buffer zone 
    
    if len(all_target_window_indices[-1]) > 0:
        simulation_steps_needed = int(all_target_window_indices[-1][-1] + 1 )
        # i.e. the last index in the last target data point
        # add one to account for the zero-indexing
    else:
        # if there are no target data points, then we need to use the last index in the last input data point
        # add one to account for the zero-indexing
        simulation_steps_needed = int(all_input_window_indices[-1][-1] + 1)

    # check if raw Lorenz data exists for the given params; otherwise generate it 
    valid_existing_simulation = False 

    # check params of existing data by iterating over everything in the json data directory
    if os.path.exists(DATA_DIRECTORY_PATH):
        with open(DATA_DIRECTORY_PATH, 'r') as f:
            data_directory = json.load(f)
        for entry in data_directory:
            # check if the params match 
            if (entry["K"] == K) and (entry["F"] == F) and (entry["c"] == c) and (entry["b"] == b) and (entry["h"] == h) and (entry["n_steps"] >= simulation_steps_needed) and (entry["resolution"] == time_resolution) and (entry["seed"] == seed):
                # match; get the path to the data so we can load it later 
                valid_existing_simulation = True 
                lorenz_data_path = entry["fname"]
                break 

    # otherwise, generate Lorenz data 
    if not valid_existing_simulation:
        lorenz_data_path = f"/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/data/test_{datetime.now()}.npz" # TODO update what path to save it to 
        run_download_lorenz96_2coupled(
            fname=lorenz_data_path, 
            K=K,
            F=F,
            c=c,
            b=b,
            h=h,
            n_steps=simulation_steps_needed,
            resolution=time_resolution,
            seed=seed)

    # load raw Lorenz data 
    t, X = load_lorenz96_2coupled(lorenz_data_path)
    # t has shape (simulation_steps_needed,)
    # X has shape (simulation_steps_needed, K*2)

    # iterate over windows of input/target data and convert into a series of GraphTuple objects 
    # note that the indices include the buffer section, so we first drop that
    all_input_window_indices = all_input_window_indices[init_buffer_samples:]
    all_target_window_indices = all_target_window_indices[init_buffer_samples:]
    
    input_windows = []
    target_windows = []
    for input_window_indices, target_window_indices in zip(all_input_window_indices, all_target_window_indices):
        # grab the window of data 
        input_X = X[input_window_indices] # shape (input_steps, K*2)
        target_X = X[target_window_indices] # shape (output_steps, K*2)

        # convert features into a GraphsTuple structure 
        input_graphtuples = []
        target_graphtuples = []
        for step in range(input_steps):
            # take timeslice in the window and resize to have shape (K, num_fts)
            data = np.vstack((input_X[step, :K], input_X[step, K:])).T
            graphtuple = timestep_to_graphstuple(data, K, fully_connected_edges)
            input_graphtuples.append(graphtuple)

        for step in range(output_steps):
            # take timeslice in the window and resize to have shape (K, num_fts)
            data = np.vstack((target_X[step, :K], target_X[step, K:])).T
            graphtuple = timestep_to_graphstuple(data, K, fully_connected_edges)
            target_graphtuples.append(graphtuple)

        input_windows.append(input_graphtuples)
        target_windows.append(target_graphtuples)
        
    # partition series of windows into train/val/test 
    train_upper_index = round(train_pct * n_samples)
    val_upper_index = round((train_pct + val_pct) * n_samples)

    graph_tuple_dict = {
        'train': {
            'inputs': input_windows[:train_upper_index], 
            'targets': target_windows[:train_upper_index]}, 
        'val': {
            'inputs': input_windows[train_upper_index:val_upper_index], 
            'targets': target_windows[train_upper_index:val_upper_index]}, 
        'test': {
            'inputs': input_windows[val_upper_index:], 
            'targets': target_windows[val_upper_index:]
        }}
    # type: Dict[str, Dict[str, List[List[jraph.GraphsTuple]]]]
    
    # normalize data 
    if normalize:
        graph_tuple_dict = normalize_lorenz96_2coupled(graph_tuple_dict)

    return graph_tuple_dict


@partial(jax.jit, static_argnames=["K", "fully_connected_edges"])
def timestep_to_graphstuple(data, K, fully_connected_edges):
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

    if fully_connected_edges:
        n_edges = K * K
    # if the graph is fully connected, then each edge feature indicates the shortest distance (and direction) between the sender and receiver node. 
        # since there are 35 nodes besides the sender node, we say that the nodes are split evenly between the 17 to the "right"/positive and 17 to the "left"/negative side of the sender node, with the last node arbitrarily placed on the right side of the sender. 
        for i in range(K):
            for j in range(K):
                senders += [i]
                receivers += [j]
                dist = i - j
                if dist < -17: dist += 36 # wrap around 
                elif dist > 18: dist -= 36 # wrap around 
                edge_fts += [[dist]]

                # ranged from -35 to +35 

                # we want the +35 to be -1 and -35 to be +1
                # TODO PICK UP HERE 

    else: # only have edges to the nearest and second nearest neighbors (5 total)
        n_edges = K * 5

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
        n_edge=jnp.array([n_edges]))


def print_graph_fts(graph: jraph.GraphsTuple):
    print(f'Number of nodes: {graph.n_node[0]}')
    print(f'Number of edges: {graph.n_edge[0]}')
    print(f'Node features shape: {graph.nodes.shape}')
    print(f'Edge features shape: {graph.edges.shape}')    
    print(f'Global features shape: {graph.globals.shape}')



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