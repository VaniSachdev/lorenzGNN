import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk
import functools
import optax
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
from datetime import datetime
import logging
import pdb

from utils.checkpoints import Checkpoint
from utils.logging import log_training_performance, plot_training_performance

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

    logging.debug('preds.shape {}'.format(preds.shape))
    logging.debug('targets.shape{}'.format(targets.shape))

    # MSE loss
    loss = jnp.mean(jnp.square(preds - targets))

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
        logging.debug('in rollout_step')
        pred = model.apply(params, input_graph)
        return pred

    x = input_graph

    f = functools.partial(rollout_step, params=params, model=model)
    preds = jax.lax.scan(f, x)

    # preds = jax.lax.fori_loop(lower=0, upper=n_rollout_steps, body_fun=rollout_step, init_val=(params, x, model))

    logging.debug(type(preds))
    logging.debug(len(preds))
    logging.debug(type(preds[0]))

    return preds


def train(net_fn, data_dict_lists: Dict[str, List[Dict[str, Any]]], epochs: int,
          cfg: Dict) -> hk.Params:
    """Training loop.  
    
        Args:
            net_fn (function): jax model function
            data_dict_lists (dict): dictionary of train, val, and test datasets
            epochs (int): number of epochs to train for
            cfg (dict): yaml config dictionary

        Returns:
            model params of the trained model
    """
    logging.info('\n--------------------\n')
    logging.info('Training loop')

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate graph and label to initialize the network.
    train_dataset = data_dict_lists["train"]
    val_dataset = data_dict_lists["val"]
    graph = train_dataset[0]['input_graph']
    print(type(graph))
    pdb.set_trace()

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

    # for now, training proceeds one graph at a time
    # TODO: implement batch training
    for epoch in range(epochs):
        start = datetime.now()
        accumulated_loss = 0

        for batch_idx in range(len(train_dataset)):
            graph = train_dataset[batch_idx]['input_graph']
            target = train_dataset[batch_idx]['target'][
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

            # compute loss and update weights
            (loss, pred), grad = compute_loss_fn(params,
                                                 input_graph=graph,
                                                 target_graph=target)
            updates, opt_state = opt_update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            # save checkpoint
            checkpoint = Checkpoint(epoch=epoch,
                                    net_fn=net_fn,
                                    model_params=params,
                                    opt_state=opt_state,
                                    cfg=cfg)
            checkpoint.save_checkpoint()

            # update stats over the batch
            accumulated_loss += loss

        # update stats over the epoch
        avg_train_loss = accumulated_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        log_training_performance(cfg, epoch, "train", avg_train_loss)
        plot_training_performance(cfg)

        # assess validation performance at the end of every epoch
        avg_val_loss, val_preds = evaluate(net_fn=net_fn,
                                       val_dataset=val_dataset,
                                       params=params)
        val_losses.append(avg_val_loss)
        log_training_performance(cfg, epoch, "val", avg_val_loss)
        plot_training_performance(cfg)

        if epoch % 1 == 0:
            logging.info(f'epoch {epoch}, runtime {datetime.now() - start}')
            logging.info(f'(MSE) train loss: {avg_train_loss}, val_loss: {avg_val_loss}\n')

    return params


def evaluate(net_fn, val_dataset: List[Dict[str, Any]],
             params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script."""
    logging.info('\n--------------------\n')
    logging.info('Evaluation loop')

    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.without_apply_rng(hk.transform(net_fn))
    # Get a candidate gra               ph and label to initialize the network.
    # TODO: why do we need this again? 
    # graph = dataset[0]['input_graph']
    accumulated_loss = 0

    compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
    preds = []

    for idx in range(len(val_dataset)):
        graph = val_dataset[idx]['input_graph']
        target = val_dataset[idx]['target'][0]
        # graph = pad_graph_to_nearest_power_of_two(graph)
        # label = jnp.concatenate([label, jnp.array([0])])
        loss, pred = compute_loss_fn(params, graph, target)
        accumulated_loss += loss
        preds.append(pred)
        if idx % 1000 == 0:
            logging.info(f'Evaluated {idx + 1} graphs')
    logging.info('Completed evaluation.')
    avg_loss = 0 if accumulated_loss == 0 else accumulated_loss / len(val_dataset)
    return avg_loss, preds
