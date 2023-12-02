import jraph
import jax
import jax.numpy as jnp
import networkx as nx
from flax import linen as nn
from utils.jraph_data import print_graph_fts

from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Sequence
import pdb 


# TODO fix either here or at call to make sure we are handling the window lists 
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
        for size in self.feature_sizes[:-1]:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, 
                           deterministic=self.deterministic)(x)
        
        # we don't want an activation function like relu on the last layer 
        x = nn.Dense(features=self.feature_sizes[-1])(x)
        x = nn.Dropout(rate=self.dropout_rate, 
                        deterministic=self.deterministic)(x)

        return x


class MLPBlock(nn.Module):
    """ A single Graph Network block containing MLP functions. 
    
        Modified from Flax GN example code. 
    """
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = False 
    deterministic: bool = True
    randvar: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    edge_features: Sequence[int] = (4, 8) # the last feature size will be the number of features that the graph predicts
    node_features: Sequence[int] = (32, 2)
    global_features: Sequence[int] = None
    

    @nn.compact
    def __call__(self, 
                 input_window_graphs: Iterable[jraph.GraphsTuple],
                 ) -> jraph.GraphsTuple:
        # since we can't process input time series yet, this is just a dummy placeholder step where we select the first graph in the input window to make it workable with this model architecture
        # TODO: eventually, implement time series input
        input_graph = input_window_graphs[0]

        if self.edge_features is not None:
            update_edge_fn = jraph.concatenated_args(
                MLP(
                    feature_sizes=self.edge_features, # arbitrarily chosen for now
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    activation=self.activation,
                )
            )
        else:
            update_edge_fn = None 

        if self.node_features is not None:
            update_node_fn = jraph.concatenated_args(
                MLP(
                    feature_sizes=self.node_features, # arbitrarily chosen for now. we want the last layer to be 2 so that we get the same number of node features that we put in. 
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    activation=self.activation,
                )
            )
        else:
            update_node_fn = None 

        if self.global_features is not None:
            update_global_fn = jraph.concatenated_args(
                MLP(
                    feature_sizes=self.global_features,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    activation=self.activation,
                )
            )
        else:
            update_global_fn = None 

        graph_net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=update_global_fn,
        )

        processed_graphs = graph_net(input_graph)
        # revert edge features to their original values
        # we want the edges to be encoded/processed by the update_edge_fn internally as part of the processing for the node features, but we only use the encoded edges internally and don't want it to affect the actual graph structure of the data because we know that it is fixed 
        processed_graphs = processed_graphs._replace(edges=input_graph.edges)

        if self.skip_connections:
            processed_graphs = add_graphs_tuples_nodes(processed_graphs, input_graph)

        if self.layer_norm:
            # TODO: why does layernorm cause the edge features to all be 0? 
            processed_graphs = processed_graphs._replace(
                nodes=nn.LayerNorm()(processed_graphs.nodes),
                edges=nn.LayerNorm()(processed_graphs.edges),
                globals=nn.LayerNorm()(processed_graphs.globals),
            )

        
        return [processed_graphs] # so that the input and output types will be consistent, and allow nn.Sequential to work
    

class MLPGraphNetwork(nn.Module):
    """ A complete Graph Network core consisting of a sequence of MLPBlocks. 
    """
    n_blocks: int # i.e. number of message-passing steps if params are shared
    share_params: bool # whether iterated blocks should be identical or distinct
    dropout_rate: float = 0
    skip_connections: bool = True
    layer_norm: bool = False
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    edge_features: Sequence[int] = (4, 8) # the last feature size will be the number of features that the graph predicts
    node_features: Sequence[int] = (32, 2)
    global_features: Sequence[int] = None

    @nn.compact
    def __call__(
        self, 
        input_window_graphs: Iterable[jraph.GraphsTuple],
    ) -> jraph.GraphsTuple:
        # since we can't process input time series yet, this is just a dummy placeholder step where we select the first graph in the input window to make it workable with this model architecture
        # TODO: eventually, implement time series input
        assert self.n_blocks > 0

        blocks = []
        # TODO: should we be defining blocks here or in some kind of init/setup function ?

        if self.share_params:
            shared_block = MLPBlock(
                dropout_rate=self.dropout_rate,
                skip_connections = self.skip_connections,
                layer_norm = self.layer_norm,
                deterministic = self.deterministic,
                edge_features=self.edge_features,      
                node_features=self.node_features,      
                global_features=self.global_features,   
                activation=self.activation,   
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
                    deterministic = self.deterministic,      
                    edge_features=self.edge_features,      
                    node_features=self.node_features,      
                    global_features=self.global_features,      
                    activation=self.activation,   
                ))
                # TODO: check that this create distinct blocks with unshared params

        # Apply a Graph Network once for each message-passing round.
        processed_graphs_list = nn.Sequential(blocks)(input_window_graphs)
        # TODO: do we need skip connections or layer_norm here? 

        return processed_graphs_list


def naive_const_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return graph


def naive_zero_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    zeros = jnp.zeros_like(graph.nodes)
    return graph._replace(nodes=zeros)
