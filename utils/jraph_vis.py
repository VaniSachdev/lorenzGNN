import numpy as np

import matplotlib.pyplot as plt

import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

from utils.jraph_data import convert_jraph_to_networkx_graph


# graph visualizations
def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)

    pos = nx.circular_layout(nx_graph)
    X1_attr = [nx_graph.nodes[i]['node_feature'][0] for i in nx_graph.nodes]

    nx.draw_networkx(nx_graph,
                     pos=pos,
                     cmap=plt.cm.RdBu,
                     node_color=X1_attr,
                     vmin=-8,
                     vmax=8,
                     with_labels=True,
                     node_size=100,
                     font_size=6,
                     font_color='black')

    # get colorbar
    ax = plt.gca()
    PCM = ax.get_children()[0]  # this index may vary
    plt.colorbar(PCM, ax=ax)
    plt.title('X1 data for nodes')


# time series visualizations
def plot_time_series_for_node(graph_tuple_dict, node):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle("sampled time series after reshaping", size=28)
    ax0.set_title("X1 (i.e. atmospheric variable) for node {}".format(node),
                  size=20)
    ax1.set_title("X2 (i.e. oceanic variable) for node {}".format(node),
                  size=20)
    plt.xlabel('time (days)', size=16)

    # plot train data
    ax0.plot(jnp.array([g.nodes[node][0] for g in graph_tuple_dict['train']]),
             label='train')
    ax1.plot(jnp.array([g.nodes[node][1] for g in graph_tuple_dict['train']]),
             label='train')

    # plot val data
    ax0.plot(range(
        len(graph_tuple_dict['train']),
        len(graph_tuple_dict['train']) + len(graph_tuple_dict['val'])),
             jnp.array([g.nodes[node][0] for g in graph_tuple_dict['val']]),
             label='val')
    ax1.plot(range(
        len(graph_tuple_dict['train']),
        len(graph_tuple_dict['train']) + len(graph_tuple_dict['val'])),
             jnp.array([g.nodes[node][1] for g in graph_tuple_dict['val']]),
             label='val')

    ax0.legend()
    ax1.legend()


def plot_rollout_for_node(data_dict_list,
                          timestep_duration,
                          n_rollout_steps,
                          node,
                          title=''):

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 8))
    if title == '':
        fig.suptitle(f"rollout for node {node}", size=28)
    else:
        fig.suptitle(title, size=28)
    ax0.set_title("X1 (i.e. atmospheric variable) for node {}".format(node),
                  size=20)
    ax1.set_title("X2 (i.e. oceanic variable) for node {}".format(node),
                  size=20)
    plt.xlabel('time (days)', size=16)

    inputs_x1 = [
        data_dict['input_graph'].nodes[node][0] for data_dict in data_dict_list
    ]
    inputs_x2 = [
        data_dict['input_graph'].nodes[node][1] for data_dict in data_dict_list
    ]

    targets_x1 = jnp.array(
        np.ravel([[
            data_dict['target'][i].nodes[node][0]
            for i in range(n_rollout_steps)
        ] for data_dict in data_dict_list]))
    targets_x2 = jnp.array(
        np.ravel([[
            data_dict['target'][i].nodes[node][1]
            for i in range(n_rollout_steps)
        ] for data_dict in data_dict_list]))

    targets_t = jnp.array(
        np.ravel([[(t + i * timestep_duration)
                   for i in range(1, n_rollout_steps + 1)]
                  for t in range(len(data_dict_list))]))

    # plot inputs
    ax0.plot(inputs_x1, alpha=0.4, label='input', c='blue')
    ax1.plot(inputs_x2, alpha=0.4, label='input', c='blue')
    # plot rollout targets
    ax0.scatter(targets_t, targets_x1, alpha=0.4, label='targets', c='green')
    ax1.scatter(targets_t, targets_x2, alpha=0.4, label='targets', c='green')

    ax0.legend()
    ax1.legend()


def plot_predictions(
        data_dict_list,
        preds,
        timestep_duration,
        n_rollout_steps,
        #  total_steps,
        node,
        title=''):

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 8))
    if title == '':
        fig.suptitle(f"predictions for node {node}", size=28)
    else:
        fig.suptitle(title, size=28)
    ax0.set_title("X1 (i.e. atmospheric variable) for node {}".format(node),
                  size=20)
    ax1.set_title("X2 (i.e. oceanic variable) for node {}".format(node),
                  size=20)
    plt.xlabel('time (days)', size=16)

    inputs_x1 = [
        data_dict['input_graph'].nodes[node][0] for data_dict in data_dict_list
    ]
    inputs_x2 = [
        data_dict['input_graph'].nodes[node][1] for data_dict in data_dict_list
    ]

    preds_x1 = [pred[node][0] for pred in preds]
    preds_x2 = [pred[node][1] for pred in preds]

    targets_x1 = jnp.array(
        np.ravel([[
            data_dict['target'][i].nodes[node][0]
            for i in range(n_rollout_steps)
        ] for data_dict in data_dict_list]))
    targets_x2 = jnp.array(
        np.ravel([[
            data_dict['target'][i].nodes[node][1]
            for i in range(n_rollout_steps)
        ] for data_dict in data_dict_list]))

    targets_t = jnp.array(
        np.ravel([[(t + i * timestep_duration)
                   for i in range(1, n_rollout_steps + 1)]
                  for t in range(len(data_dict_list))]))

    # plot inputs
    ax0.plot(inputs_x1, alpha=0.8, label='input', c='blue')
    ax1.plot(inputs_x2, alpha=0.8, label='input', c='blue')

    # plot rollout targets
    ax0.scatter(
        targets_t,
        targets_x1,
        # s=5,
        alpha=0.8,
        label='targets',
        c='green')
    ax1.scatter(targets_t, targets_x2, alpha=0.8, label='targets', c='green')

    # plot predictions
    ax0.scatter(
        range(timestep_duration, timestep_duration + len(data_dict_list)),
        preds_x1,
        # s=5,
        alpha=0.8,
        label='preds',
        c='orange')
    ax1.scatter(
        range(timestep_duration, timestep_duration + len(data_dict_list)),
        preds_x2,
        # s=5,
        alpha=0.8,
        label='preds',
        c='orange')

    ax0.legend()
    ax1.legend()