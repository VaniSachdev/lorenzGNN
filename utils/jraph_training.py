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

# def pred_loss(pred_graph: jraph.GraphsTuple, target_graph: jraph.GraphsTuple):
#     """ Compute MSE loss between an already-given predicted graph and its target.
#     """
#     preds = pred_graph.nodes
#     targets = target_graph.nodes

#     loss = jnp.mean(jnp.square(preds - targets))

#     print(loss.shape)
#     return loss, preds


def compute_loss(params: hk.Params, input_graph: jraph.GraphsTuple,
                 target_graph: jraph.GraphsTuple,
                 net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes loss for a one-step prediction (no rollout). 

        Distinct from pred_loss since this applies a model to an input_graph
    
        Also returns predicted nodes.
    """
    pred_graph = net.apply(params, input_graph)

    preds = pred_graph.nodes
    targets = target_graph.nodes

    print('preds.shape', preds.shape)
    print('targets.shape', targets.shape)

    # MSE loss
    loss = jnp.mean(jnp.square(preds - targets))

    print(loss.shape)
    return loss, preds


# TODO: DO NOT USE YET, THIS IS THROWING MAJOR ERRORS
# NEED TO FIX THE JAX.LAX SCAN/FORI_LOOP!
def rollout(
        params: hk.Params,
        input_graph: jraph.GraphsTuple,
        #  target_graphs: Iterable[jraph.GraphsTuple],
        n_rollout_steps: int,
        model: jraph.GraphsTuple):

    def rollout_step(params: hk.Params, input_graph: jraph.GraphsTuple,
                     model: jraph.GraphsTuple):
        print('in rollout_step')
        pred = model.apply(params, input_graph)
        return pred

    x = input_graph

    f = functools.partial(rollout_step, params=params, model=model)
    preds = jax.lax.scan(f, x)

    # preds = jax.lax.fori_loop(lower=0, upper=n_rollout_steps, body_fun=rollout_step, init_val=(params, x, model))

    print(type(preds))
    print(len(preds))
    print(type(preds[0]))

    return preds


def train(net_fn, dataset: List[Dict[str, Any]],
          num_train_steps: int) -> hk.Params:
    """Training loop."""
    # TODO: change num_train_steps to epochs

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph = dataset[0]['input_graph']

    # Initialize the network.
    params = net.init(jax.random.PRNGKey(42), graph)
    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(1e-4)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

    train_losses = []
    val_losses = []

    for idx in range(num_train_steps):
        # epochs = num_train_steps / len(dataset)
        graph = dataset[idx % len(dataset)]['input_graph']
        target = dataset[idx % len(dataset)]['target'][
            0]  # get first data graph in the rollout for now

        # Jax will re-jit your graphnet every time a new graph shape is encountered.
        # In the limit, this means a new compilation every training step, which
        # will result in *extremely* slow training. To prevent this, pad each
        # batch of graphs to the nearest power of two. Since jax maintains a cache
        # of compiled programs, the compilation cost is amortized.
        # graph = pad_graph_to_nearest_power_of_two(graph)

        # Since padding is implemented with pad_with_graphs, an extra graph has
        # been added to the batch, which means there should be an extra label.
        # label = jnp.concatenate([label, jnp.array([0])])

        (loss, pred), grad = compute_loss_fn(params,
                                             input_graph=graph,
                                             target_graph=target)
        # TODO: ADD VAL PREDS AND LOSSES 
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if idx % 50 == 0:
            print(f'step: {idx}, loss: {loss}, pred: {type(pred)}')
    print('Training finished')
    return params


def evaluate(net_fn, dataset: List[Dict[str, Any]],
             params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script."""
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    graph = dataset[0]['input_graph']
    accumulated_loss = 0
    #   accumulated_accuracy = 0
    compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
    preds = []

    for idx in range(len(dataset)):
        graph = dataset[idx]['input_graph']
        target = dataset[idx]['target'][0]
        # graph = pad_graph_to_nearest_power_of_two(graph)
        # label = jnp.concatenate([label, jnp.array([0])])
        loss, pred = compute_loss_fn(params, graph, target)
        # accumulated_accuracy += acc
        accumulated_loss += loss
        preds.append(pred)
        if idx % 100 == 0:
            print(f'Evaluated {idx + 1} graphs')
    print('Completed evaluation.')
    loss = accumulated_loss / idx
    #   accuracy = accumulated_accuracy / idx
    print(f'Eval loss: {loss}')
    return loss, preds
