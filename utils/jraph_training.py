import jraph
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training import train_state # Simple train state for the common case with a single Optax optimizer.
from clu import metrics 
import optax

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
from datetime import datetime
import logging
import pdb
import os 
from copy import deepcopy

from utils.checkpoints import Checkpoint
from utils.logging import log_training_performance, plot_training_performance
from utils.jraph_models import MLPBlock, MLPGraphNetwork


def MSE(targets, preds):
    mse = jnp.mean(jnp.square(preds - targets))
    return mse 

def one_step_loss(state: train_state.TrainState, 
                  input_graph: jraph.GraphsTuple,
                 target_graph: jraph.GraphsTuple,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Computes loss for a one-step prediction (no rollout). 
    
        Also returns predicted nodes.
    """
    pred_graph = state.apply_fn(state.params, input_graph) 
    preds = pred_graph.nodes
    targets = target_graph.nodes

    # MSE loss
    loss = MSE(targets, preds)
    return loss, preds

def rollout_loss(state: train_state.TrainState, 
                  input_graph: jraph.GraphsTuple,
                 target_graphs: Iterable[jraph.GraphsTuple], # i dont think this can be a jnp array, will that be a problem? 
                 n_steps: int,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Computes average loss for an n-step rollout. 
    
        Also returns predicted nodes.
    """
    # TODO: theoretically n_steps can be eliminated and we just base the rollout on the size of the target_graphs list 
    assert n_steps > 0
    assert len(target_graphs) == n_steps, (len(target_graphs), n_steps)

    curr_input_graph = input_graph
    pred_nodes = []
    total_loss = 0
    for i in range(n_steps):
        pred_graph = state.apply_fn(state.params, curr_input_graph) 
        curr_input_graph = pred_graph

        preds = pred_graph.nodes
        targets = target_graphs[i].nodes
        loss = MSE(targets, preds)

        pred_nodes.append(preds) # Side-effects aren't allowed in JAX-transformed functions, and appending to a list is a side effect ??
        total_loss += loss

    avg_loss = total_loss / n_steps

    return avg_loss, pred_nodes


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')
#   rollout_loss: metrics.Average.from_fun(rollout_loss)

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')
#   loss: metrics.Average.from_fun(one_step_loss)

# @jax.jit
def train_step_fn(
    state: train_state.TrainState,
    input_graph: jraph.GraphsTuple,
    target_graphs: Iterable[jraph.GraphsTuple],
    n_steps: int = 1,
    # rngs: Dict[str, jnp.ndarray] = None, # TODO: add rngs arg to state.apply_fn later
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, input_graph, target_graphs):
        curr_state = state.replace(params=params) # create a new state object so that we can pass the whole thing into the one_step_loss function. we do this so that we can keep track of the original state's apply_fn() and a custom param together (theoretically the param argument in this function doesn't need to be the same as the default state's param)
        loss, pred_nodes = rollout_loss(curr_state, input_graph, target_graphs, n_steps)
        return loss, pred_nodes
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred_nodes), grads = grad_fn(state.params, input_graph, target_graphs)
    state = state.apply_gradients(grads=grads) # update params in the state 

    metrics_update = TrainMetrics.single_from_model_output(loss=loss)
    # theoretically seems like could also pass preds directly to trainmetrics to compute the loss there 
    # is one better than another? 
    return state, metrics_update, pred_nodes
    
train_step = jax.jit(train_step_fn, static_argnames=["n_steps"])

def train_and_evaluate(net, data_dict_lists: Dict[str, List[Dict[str, Any]]], epochs: int, learning_rate, 
          cfg: Dict
    ) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the TensorBoard summaries are written to.

    Returns:
        The train state (which includes the `.params`).
    """
    # We only support single-host training.
    assert jax.process_count() == 1

    # # Create writer for logs.
    # writer = metric_writers.create_default_writer(workdir)
    # writer.write_hparams(config.to_dict())

    # # Get datasets, organized by split.
    # logging.info('Obtaining datasets.')
    # datasets = input_pipeline.get_datasets(
    #     config.batch_size,
    #     add_virtual_node=config.add_virtual_node,
    #     add_undirected_edges=config.add_undirected_edges,
    #     add_self_loops=config.add_self_loops,
    # )
    # train_iter = iter(datasets['train'])

    # Create and initialize the network.
    logging.info('Initializing network.')
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)

    train_dataset = data_dict_lists["train"]
    val_dataset = data_dict_lists["val"]
    init_graphs = train_dataset[0]['input_graph']
    # init_graphs = replace_globals(init_graphs)
    # init_net = create_model(config, deterministic=True)
    # TODO: why did they have an init_net separate from the actual net? 
    params = jax.jit(net.init)(init_rng, init_graphs)
    # parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = optax.adam(learning_rate=learning_rate)
    # tx = create_optimizer(config)

    # Create the training state.
    # net = create_model(config, deterministic=False)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx
    )

    # Set up checkpointing of the model.
    # The input pipeline cannot be checkpointed in its current form,
    # due to the use of stateful operations.
    # TODO implement checkpoint later
    # checkpoint_dir = os.path.join(workdir, 'checkpoints')
    # ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
    # state = ckpt.restore_or_initialize(state)
    # initial_step = int(state.step) + 1
    initial_step = 1

    # Create the evaluation state, corresponding to a deterministic model.
    # TODO: why is eval_net separate from net? 
    eval_net = deepcopy(net)
    # eval_net = create_model(config, deterministic=True)
    eval_state = state.replace(apply_fn=eval_net.apply)

    # # Hooks called periodically during training.
    # report_progress = periodic_actions.ReportProgress(
    #     num_train_steps=config.num_train_steps, writer=writer
    # )
    # profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    # hooks = [report_progress, profiler]

    # Begin training loop.
    logging.info('Starting training.')
    train_metrics = None
    batch_size = 0
    n_steps = batch_size * epochs
    for step in range(initial_step, n_steps + 1):
        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
            # TODO: is this by batch? iter()/next is only single element but the dataset was constructed with batches 
            # TODO: BROKEN, must separately identify input and output graphs 
            graphs = jax.tree_util.tree_map(np.asarray, next(train_iter))
            state, metrics_update = train_step(
                state, graphs, rngs={'dropout': dropout_rng}
            )

            # Update metrics.
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 10, step)
        # for hook in hooks:
        #     hook(step)

        # # Log, if required.
        # is_last_step = step == config.num_train_steps - 1
        # if step % config.log_every_steps == 0 or is_last_step:
        #     writer.write_scalars(
        #         step, add_prefix_to_keys(train_metrics.compute(), 'train')
        #     )
        #     train_metrics = None

        # # Evaluate on validation and test splits, if required.
        # if step % config.eval_every_steps == 0 or is_last_step:
        #     eval_state = eval_state.replace(params=state.params)

        #     splits = ['validation', 'test']
        #     with report_progress.timed('eval'):
        #         eval_metrics = evaluate_model(eval_state, datasets, splits=splits)
        #     for split in splits:
        #         writer.write_scalars(
        #             step, add_prefix_to_keys(eval_metrics[split].compute(), split)
        #         )

        # Checkpoint model, if required.
        # if step % config.checkpoint_every_steps == 0 or is_last_step:
        #     with report_progress.timed('checkpoint'):
        #       ckpt.save(state)

    return state

