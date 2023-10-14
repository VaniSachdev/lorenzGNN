import jraph
import jax
import jax.numpy as jnp
import networkx as nx
from flax import linen as nn

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Sequence

from utils.jraph_data import convert_jraph_to_networkx_graph


def add_graphs_tuples_nodes(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
  """Adds only the node features from other_graphs to graphs."""
  return graphs._replace(
      nodes=graphs.nodes + other_graphs.nodes,
      edges=graphs.edges,
      globals=graphs.globals,
  )


class MLP(nn.Module):
    """ A multi-layer perceptron.
    
        Copied from Flax example models. 
    """

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # TODO: would be nice if we could set a custom name for the module 

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(
                x
            )
        return x


class MLPBlock(nn.Module):
    """ A single Graph Network block containing MLP functions. 
    
        Modified from Flax GN example code. 
    """
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = False 
    deterministic: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # TODO: do we actually not want these? 
        update_edge_fn = jraph.concatenated_args(
            MLP(
                feature_sizes=[16, 8], # arbitrarily chosen for now
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
            )
        )

        update_node_fn = jraph.concatenated_args(
            MLP(
                feature_sizes=[32, 2], # arbitrarily chosen for now. we want the last layer to be 2 so that we get the same number of node features that we put in. 
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
            )
        )
        # update_global_fn = jraph.concatenated_args(
        #     MLP(
        #         feature_sizes=[1],
        #         dropout_rate=self.dropout_rate,
        #         deterministic=self.deterministic,
        #     )
        # )

        graph_net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=None,
        )

        processed_graphs = graph_net(graphs)
        # revert edge features to their original values
        # we want the edges to be encoded/processed by the update_edge_fn internally as part of the processing for the node features, but we only use the encoded edges internally and don't want it to affect the actual graph structure of the data because we know that it is fixed 
        processed_graphs = processed_graphs._replace(edges=graphs.edges)

        if self.skip_connections:
            processed_graphs = add_graphs_tuples_nodes(processed_graphs, graphs)

        if self.layer_norm:
            # TODO: why does layernorm cause the edge features to all be 0? 
            processed_graphs = processed_graphs._replace(
                nodes=nn.LayerNorm()(processed_graphs.nodes),
                edges=nn.LayerNorm()(processed_graphs.edges),
                globals=nn.LayerNorm()(processed_graphs.globals),
            )

        return processed_graphs
    

class MLPGraphNetwork(nn.Module):
    """ A complete Graph Network core consisting of a sequence of MLPBlocks. 
    """
    n_blocks: int # i.e. number of message-passing steps if params are shared
    share_params: bool # whether iterated blocks should be identical or distinct
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = False
    deterministic: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        blocks = []
        # TODO: should we be defining blocks here or in some kind of init/setup function ?

        if self.share_params:
            shared_block = MLPBlock(
                dropout_rate=self.dropout_rate,
                skip_connections = self.skip_connections,
                layer_norm = self.layer_norm,
                deterministic = self.deterministic           
            )
            for _ in range(self.n_blocks):
                blocks.append(shared_block)
                # TODO: i have no idea if this will actually work i.e. will it be recurrent or not. would need to check size of params to verify
        else:
            for _ in range(self.n_blocks):
                blocks.append(MLPBlock(
                    dropout_rate=self.dropout_rate,
                    skip_connections = self.skip_connections,
                    layer_norm = self.layer_norm,
                    deterministic = self.deterministic           
                ))
                # TODO: check that this create distinct blocks with unshared params

        # Apply a Graph Network once for each message-passing round.
        processed_graphs = nn.Sequential(blocks)(graphs)
        # TODO: do we need skip connections or layer_norm here? 

        return processed_graphs



# TODO: do we still need these 
# def MLPBlock_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
#     net = MLPBlock()
#     return net(graph)


# def MLPGraphNetwork_fn(input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
#     net = MLPGraphNetwork(n_blocks=9)
#     return net(input_graph)


def naive_const_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return graph


def naive_zero_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    zeros = jnp.zeros_like(graph.nodes)
    return graph._replace(nodes=zeros)
