# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definition of the GNN model."""

from typing import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jraph


# def add_graphs_tuples(
#     graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
# ) -> jraph.GraphsTuple:
#   """Adds the nodes, edges and global features from other_graphs to graphs."""
#   return graphs._replace(
#       nodes=graphs.nodes + other_graphs.nodes,
#       edges=graphs.edges + other_graphs.edges,
#       globals=graphs.globals + other_graphs.globals,
#   )

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


def naive_const_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return graph


def naive_zero_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    zeros = jnp.zeros_like(graph.nodes)
    return graph._replace(nodes=zeros)


# TODO how does the flax example model handle batches of inputs?? 

# class GraphNet(nn.Module):
#   """A complete Graph Network model defined with Jraph."""

#   latent_size: int
#   num_mlp_layers: int
#   message_passing_steps: int
#   output_globals_size: int
#   dropout_rate: float = 0
#   skip_connections: bool = True
#   use_edge_model: bool = True
#   layer_norm: bool = True
#   deterministic: bool = True

#   @nn.compact
#   def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
#     # We will first linearly project the original features as 'embeddings'.
#     embedder = jraph.GraphMapFeatures(
#         embed_node_fn=nn.Dense(self.latent_size),
#         embed_edge_fn=nn.Dense(self.latent_size),
#         embed_global_fn=nn.Dense(self.latent_size),
#     )
#     processed_graphs = embedder(graphs)

#     # Now, we will apply a Graph Network once for each message-passing round.
#     mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
#     for _ in range(self.message_passing_steps):
#       if self.use_edge_model:
#         update_edge_fn = jraph.concatenated_args(
#             MLP(
#                 mlp_feature_sizes,
#                 dropout_rate=self.dropout_rate,
#                 deterministic=self.deterministic,
#             )
#         )
#       else:
#         update_edge_fn = None

#       update_node_fn = jraph.concatenated_args(
#           MLP(
#               mlp_feature_sizes,
#               dropout_rate=self.dropout_rate,
#               deterministic=self.deterministic,
#           )
#       )
#       update_global_fn = jraph.concatenated_args(
#           MLP(
#               mlp_feature_sizes,
#               dropout_rate=self.dropout_rate,
#               deterministic=self.deterministic,
#           )
#       )

#       graph_net = jraph.GraphNetwork(
#           update_node_fn=update_node_fn,
#           update_edge_fn=update_edge_fn,
#           update_global_fn=update_global_fn,
#       )

#       if self.skip_connections:
#         processed_graphs = add_graphs_tuples(
#             graph_net(processed_graphs), processed_graphs
#         )
#       else:
#         processed_graphs = graph_net(processed_graphs)

#       if self.layer_norm:
#         processed_graphs = processed_graphs._replace(
#             nodes=nn.LayerNorm()(processed_graphs.nodes),
#             edges=nn.LayerNorm()(processed_graphs.edges),
#             globals=nn.LayerNorm()(processed_graphs.globals),
#         )

#     # Since our graph-level predictions will be at globals, we will
#     # decode to get the required output logits.
#     decoder = jraph.GraphMapFeatures(
#         embed_global_fn=nn.Dense(self.output_globals_size)
#     )
#     processed_graphs = decoder(processed_graphs)

#     return processed_graphs


# class GraphConvNet(nn.Module):
#   """A Graph Convolution Network + Pooling model defined with Jraph."""

#   latent_size: int
#   num_mlp_layers: int
#   message_passing_steps: int
#   output_globals_size: int
#   dropout_rate: float = 0
#   skip_connections: bool = True
#   layer_norm: bool = True
#   deterministic: bool = True
#   pooling_fn: Callable[
#       [jnp.ndarray, jnp.ndarray, jnp.ndarray],  # pytype: disable=annotation-type-mismatch  # jax-ndarray
#       jnp.ndarray,
#   ] = jraph.segment_mean

#   def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
#     """Pooling operation, taken from Jraph."""

#     # Equivalent to jnp.sum(n_node), but JIT-able.
#     sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
#     # To aggregate nodes from each graph to global features,
#     # we first construct tensors that map the node to the corresponding graph.
#     # Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
#     n_graph = graphs.n_node.shape[0]
#     node_graph_indices = jnp.repeat(
#         jnp.arange(n_graph),
#         graphs.n_node,
#         axis=0,
#         total_repeat_length=sum_n_node,
#     )
#     # We use the aggregation function to pool the nodes per graph.
#     pooled = self.pooling_fn(graphs.nodes, node_graph_indices, n_graph)  # pytype: disable=wrong-arg-types  # jax-ndarray
#     return graphs._replace(globals=pooled)

#   @nn.compact
#   def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
#     # We will first linearly project the original node features as 'embeddings'.
#     embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
#     processed_graphs = embedder(graphs)

#     # Now, we will apply the GCN once for each message-passing round.
#     for _ in range(self.message_passing_steps):
#       mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
#       update_node_fn = jraph.concatenated_args(
#           MLP(
#               mlp_feature_sizes,
#               dropout_rate=self.dropout_rate,
#               deterministic=self.deterministic,
#           )
#       )
#       graph_conv = jraph.GraphConvolution(
#           update_node_fn=update_node_fn, add_self_edges=True
#       )

#       if self.skip_connections:
#         processed_graphs = add_graphs_tuples(
#             graph_conv(processed_graphs), processed_graphs
#         )
#       else:
#         processed_graphs = graph_conv(processed_graphs)

#       if self.layer_norm:
#         processed_graphs = processed_graphs._replace(
#             nodes=nn.LayerNorm()(processed_graphs.nodes),
#         )

#     # We apply the pooling operation to get a 'global' embedding.
#     processed_graphs = self.pool(processed_graphs)

#     # Now, we decode this to get the required output logits.
#     decoder = jraph.GraphMapFeatures(
#         embed_global_fn=nn.Dense(self.output_globals_size)
#     )
#     processed_graphs = decoder(processed_graphs)

#     return processed_graphs
