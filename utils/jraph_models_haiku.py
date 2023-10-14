import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

from utils.jraph_data import convert_jraph_to_networkx_graph


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Edge update function for graph net."""
    net = hk.nets.MLP(output_sizes=[36, 1])
    return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.nets.MLP(output_sizes=[36, 2])
    return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Global update function for graph net."""
    net = hk.nets.MLP(output_sizes=[1])
    return net(feats)


class MLPBlock(hk.Module):

    def __init__(
        self,
        #  edge_mlp_features: Iterable[int],
        #  node_mlp_features: Iterable[int],
        #  graph_mlp_features: Iterable[int],
        name: str = "MLPBlock"):
        """ A function that creates a single GN block with MLP edge, node, and global models, and then passes input_graph through the model to transform it

            Returns: a transformed graph

            Args:
                input_graph:
                *_mlp_features (int list):
        """
        super(MLPBlock, self).__init__(name=name)
        self._graph_net = jraph.GraphNetwork(update_node_fn=node_update_fn,
                                             update_edge_fn=None,
                                             update_global_fn=None)
        # we have to use a lambda we want to return a function, not the module itself. it we try to return the module, we will get an error that the module must be initialized inside hk.transform

    def __call__(self, input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self._graph_net(input_graph)


# we need to concat_args decorator because the GraphNetwork will pass in
# multiple arguments (e.g. edges + source features + etc etc)
# TODO: maybe turn this into a partial func? so then we can pass in custom
# output_sizes


class MLPGraphNetwork(hk.Module):
    """GraphNetwork consist of a sequence of MLPBlocks."""

    def __init__(
        self,
        n_blocks: int,
        #  edge_mlp_features:Iterable[int],
        #  node_mlp_features:Iterable[int],
        #  graph_mlp_features:Iterable[int],
        name: str = "MLPGraphNetwork"):
        """ Initializes

            Args:
                n_blocks (int): number of MLP blocks
                recurrent (bool): whether or not to share weights btwn blocks
        """
        super(MLPGraphNetwork, self).__init__(name=name)
        self.n_blocks = n_blocks
        # this feels stupid
        # TODO: this probably needs lax.scan for a for loop
        self._network = hk.Sequential([
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock(),
            MLPBlock()
        ])
        # this commented out code produces errors
        # self._network = hk.Sequential([
        #     MLPBlock(
        #         # edge_mlp_features,
        #         #  node_mlp_features,
        #         #  graph_mlp_features,
        #         name=f'Block_{i}')
        # ] for i in range(n_blocks))

    def __call__(self, input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self._network(input_graph)


def MLPBlock_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    net = MLPBlock()
    return net(graph)


def MLPGraphNetwork_fn(input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    net = MLPGraphNetwork(n_blocks=9)
    return net(input_graph)


def naive_const_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return graph


def naive_zero_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    zeros = jnp.zeros_like(graph.nodes)
    return graph._replace(nodes=zeros)
