import unittest
import logging
from datetime import datetime
from run_net import set_up_logging

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import jax.random
import flax
import jraph
import optax
from flax.training import train_state  # Simple train state for the common case with a single Optax optimizer.

from utils.jraph_models import MLPBlock, MLPGraphNetwork
from utils.jraph_training import train_step, one_step_loss


def get_dummy_graphtuple(seed=42):
    """ Create a dummy graphtuple object for testing. 
    
        The edge features are correctly set up; the node features have the 
        correct shape but are populated with random values. 
    """
    K = 36
    n_fts = 2
    rng = np.random.default_rng(seed=seed)
    dummy_data = rng.random((K, n_fts))  # array of shape (K, num_fts=2)

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
        globals=jnp.array(
            [[1.]]
        ),  # placeholder global features for now (was an empty array and None both causing errors down the line?)
        # globals=jnp.array([]),  # no global features for now
        # globals=None,  # no global features for now
        nodes=jnp.array(
            dummy_data),  # node features = state values. shape of (K, 2)
        edges=jnp.array(edge_fts, dtype=float),
        receivers=jnp.array(receivers),
        senders=jnp.array(senders),
        n_node=jnp.array([K]),
        n_edge=jnp.array([K * 5]))


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.K = 36
        self.F = 8
        self.c = 10
        self.b = 10
        self.h = 1
        self.seed = 42
        self.n_fts = 2

    def predict_helper_test(self, model):
        """ Helper function to test predictions for an arbitrary model. 
        
        """
        # get sample graph
        test_input_graph = get_dummy_graphtuple()

        # set up params
        init_graphs = test_input_graph
        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)
        params = jax.jit(model.init)(init_rng, init_graphs)

        # set up optimizer (needed for the state even if we aren't training)
        learning_rate = 0.001  # default learning rate for adam in keras
        tx = optax.adam(learning_rate=learning_rate)

        # set up state object, which helps us keep track of the model, params, and optimizer
        state = train_state.TrainState.create(apply_fn=model.apply,
                                              params=params,
                                              tx=tx)

        input_graph = test_input_graph
        pred_graph = state.apply_fn(state.params, input_graph)

        # check that the predict_helper_test shape of the node features is correct
        self.assertEqual(pred_graph.nodes.shape, test_input_graph.nodes.shape)

        # check edge features did not change
        self.assertTrue(
            jnp.array_equal(pred_graph.edges, test_input_graph.edges))

        # check global features did not change
        self.assertTrue(
            jnp.array_equal(pred_graph.globals, test_input_graph.globals))

    def train_helper(self, model, n_steps):
        """ Helper function to train arbitrary model for single step. 

            Returns initial state before training, new state after training, 
            and metrics update object. 
        """
        # get sample graph
        test_input_graph = get_dummy_graphtuple()
        test_target_graphs = [
            get_dummy_graphtuple(seed=i) for i in range(n_steps)
        ]

        # set up params
        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)
        params = jax.jit(model.init)(init_rng, test_input_graph)

        # set up optimizer
        learning_rate = 0.001  # default learning rate for adam in keras
        tx = optax.adam(learning_rate=learning_rate)

        # set up state object, which helps us keep track of the model, params, and optimizer
        init_state = train_state.TrainState.create(apply_fn=model.apply,
                                                   params=params,
                                                   tx=tx)

        # init_pred_graph = state.apply_fn(state.params, input_graph)

        # train the model for one step
        new_state, metrics_update, pred_nodes = train_step(
            init_state, test_input_graph, test_target_graphs, n_steps)

        return init_state, new_state, metrics_update, pred_nodes

    def test_predict_MLPBlock(self):
        """ test that an MLPBlock computes a single prediction correctly. """
        logging.info('\n ------------ test_predict_MLPBlock ------------ \n')
        model = MLPBlock()
        self.predict_helper_test(model)

    def test_predict_MLPGraphNetwork(self):
        """ test that an MLPGraphNetwork computes a single prediction correctly. """
        logging.info(
            '\n ------------ test_predict_MLPGraphNetwork ------------ \n')
        # test with single block, non-shared params
        model = MLPGraphNetwork(n_blocks=1, share_params=False)
        self.predict_helper_test(model)

        # TODO: tests on multi-block cores are failing
        # # test with two blocks, non-shared params
        # model = MLPGraphNetwork(n_blocks=2, share_params=False)
        # self.predict_helper_test(model)

        # # test with two blocks, shared params
        # # TODO: add tests to check that params are actually shared
        # model = MLPGraphNetwork(n_blocks=2, share_params=False)
        # self.predict_helper_test(model)

    def test_train_MLPBlock(self):
        """ test that a single train step is correctly performed on an MLPBlock. """
        logging.info('\n ------------ test_train_MLPBlock ------------ \n')
        # test for different number of rollout steps
        for n_steps in [1, 5]:
            model = MLPBlock()
            init_state, new_state, metrics_update, pred_nodes = self.train_helper(
                model, n_steps)

            # check that the state's step increased by 1 after training
            self.assertEqual(init_state.step, 0)
            self.assertEqual(new_state.step, 1)

            # check that metrics_update count was updated
            self.assertEqual(metrics_update.loss.count, 1)

            # check that the logged loss is valid
            self.assertGreater(float(metrics_update.loss.total), 0)

            # check that the number of predictions in the rollout is correct
            self.assertEqual(len(pred_nodes), n_steps)

            # check that the params are different after training
            # check params for first MLP layer in update_edge_fn
            self.assertFalse(
                jnp.array_equal(
                    init_state.params['params']['MLP_0']['Dense_0']['bias'],
                    new_state.params['params']['MLP_0']['Dense_0']['bias']))
            self.assertFalse(
                jnp.array_equal(
                    init_state.params['params']['MLP_0']['Dense_0']['kernel'],
                    new_state.params['params']['MLP_0']['Dense_0']['kernel']))

            # check params for first MLP layer in update_node_fn
            self.assertFalse(
                jnp.array_equal(
                    init_state.params['params']['MLP_1']['Dense_0']['bias'],
                    new_state.params['params']['MLP_1']['Dense_0']['bias']))
            self.assertFalse(
                jnp.array_equal(
                    init_state.params['params']['MLP_1']['Dense_0']['kernel'],
                    new_state.params['params']['MLP_1']['Dense_0']['kernel']))


if __name__ == "__main__":
    # set up logging for unittest outputs
    log_path = f"tests/outputs/model_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    set_up_logging(log_path=log_path, log_level_str="INFO")

    with open(log_path, "a") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, verbosity=2)
