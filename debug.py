# imports
import numpy as np

import matplotlib.pyplot as plt
from lorenz import lorenzDatasetWrapper
from plotters import plot_data
import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk
# for training sequence
import functools
import optax
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable


# data
def lorenz_graph(n_samples=2_000):
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
        # time_resolution=DEFAULT_TIME_RESOLUTION,
        # seed=42,
        init_buffer_steps=100,
        return_buffer=False,
        train_pct=1,
        val_pct=0,
        test_pct=0,
        override=False)
    dataset.normalize()

    # iter over time steps in lorenz data

    graph_tuple_list = []
    # construct a new data dict for each time step
    for g in dataset.train:
        # g.x has shape 36 x 2
        graph_tuple = timestep_to_graphstuple(g.x, K=K)
        graph_tuple_list.append(graph_tuple)

    # convert all data dicts into a single GraphTuple
    # graph_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)

    # return graph_tuple
    return graph_tuple_list


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
        # globals=jnp.array([]),  # no global features for now
        globals=None,  # no global features for now
        nodes=jnp.array(
            data),  # node features = atmospheric measurements. shape of (K, 2)
        edges=jnp.array(edge_fts, dtype=float),
        receivers=jnp.array(receivers),
        senders=jnp.array(senders),
        n_node=jnp.array([K]),
        n_edge=jnp.array([K * 5]))


graph_tuple_list = lorenz_graph(n_samples=200)


# model
# we need to concat_args decorator because the GraphNetwork will pass in multiple arguments (e.g. edges + source features + etc etc)
# TODO: maybe turn this into a partial func? so then we can pass in custom
# output_sizes
@jraph.concatenated_args
def update_node_MLP_fn(x: jnp.ndarray) -> jnp.ndarray:
    mlp = hk.nets.MLP(output_sizes=[16, 2])
    return mlp(x)


def MLPBlock_fn(
    input_graph: jraph.GraphsTuple
    #     , edge_mlp_features: Iterable[int],
    #  node_mlp_features: Iterable[int],
    #  graph_mlp_features: Iterable[int]
) -> jraph.GraphsTuple:
    """ A function that creates a single GN block with MLP edge, node, and global models, and then passes input_graph through the model to transform it

        Returns: a transformed graph

        Args:
            input_graph:
            *_mlp_features (int list):
    """
    graph_net = jraph.GraphNetwork(update_node_fn=update_node_MLP_fn,
                                   update_edge_fn=None,
                                   update_global_fn=None)
    # we have to use a lambda we want to return a function, not the module itself. it we try to return the module, we will get an error that the module must be initialized inside hk.transform

    return graph_net(input_graph)


# helpers
def compute_loss(input_graph: jraph.GraphsTuple,
                 target_graph: jraph.GraphsTuple, model: jraph.GraphsTuple,
                 params: hk.Params):
    """ Calculates the loss from doing an n-step rollout (i.e. the average of all losses)
    """
    # pred_graphs = []
    x = input_graph

    # TODO: check if we need anything special for this for loop
    # for i in range(n_rollout_steps):
    pred = model.apply(params, x)

    err = pred.nodes - target_graph.nodes
    mse = jnp.mean(jnp.square(err))
    return mse


if __name__ == "__main__":

    # train
    ffw_network = hk.without_apply_rng(hk.transform(MLPBlock_fn))
    # this has two functions, init and apply

    # Get a dummy graph and label to initialize the network.
    dummy_graph = graph_tuple_list[0]

    # Initialize the network.
    params = ffw_network.init(jax.random.PRNGKey(42), dummy_graph)
    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(
        1e-4)  # these are two functions to init and apply
    opt_state = opt_init(params)

    # preds = []
    # opt_state # why is this all zeros?? oh wait its the optimizer state not the weight init state

    # n_rollout = 1

    for idx in range(1):
        input_graph = graph_tuple_list[idx]
        target_graph = graph_tuple_list[idx + 1]
        # rollout_targets = graph_tuple_list[idx + 1:idx + n_rollout + 1]

        compute_loss_fn = functools.partial(compute_loss, net=ffw_network)
        compute_loss_fn = jax.jit(
            jax.value_and_grad(compute_loss_fn, has_aux=False))

        (loss, pred_graph), grad = compute_loss_fn(input_graph,
                                                   target_graph,
                                                   params=params)
    print(loss)
    print(pred_graph)
    print(grad)
