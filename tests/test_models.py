import unittest
import logging
from datetime import datetime
from run_net import set_up_logging

import jax.numpy as jnp

from utils.jraph_models import MLPBlock, MLPGraphNetwork
from tests.helpers import get_sample_data, state_setup_helper

class ModelTests(unittest.TestCase):

    def setUp(self):
        self.K = 36
        self.F = 8
        self.c = 10
        self.b = 10
        self.h = 1
        self.seed = 42
        self.n_fts = 2


    def forward_pass_helper_test(self, model):
        """ Helper function to test the forward pass for an arbitrary model. 
        
        """
        # get sample graph
        sample_dataset, _ = get_sample_data()
        sample_input_window = sample_dataset['train']['inputs'][0]
        sample_target_window = sample_dataset['train']['targets'][0]

        # set up state object, which helps us keep track of the model, params, and optimizer
        state = state_setup_helper(model)

        pred_graphs_list = state.apply_fn(state.params, sample_input_window)
        pred_graph = pred_graphs_list[0]
        first_target_graph = sample_target_window[0]

        # check that the forward_pass_helper_test shape of the node features is correct
        self.assertEqual(pred_graph.nodes.shape, first_target_graph.nodes.shape)

        # check edge features did not change
        self.assertTrue(
            jnp.array_equal(pred_graph.edges, first_target_graph.edges))

        # check global features did not change
        self.assertTrue(
            jnp.array_equal(pred_graph.globals, first_target_graph.globals))

    def test_predict_MLPBlock(self):
        """ test that an MLPBlock computes a single prediction correctly. """
        logging.info('\n ------------ test_predict_MLPBlock ------------ \n')
        model = MLPBlock()
        self.forward_pass_helper_test(model)

    def test_predict_MLPGraphNetwork(self):
        """ test that an MLPGraphNetwork computes a single prediction correctly. """
        logging.info(
            '\n ------------ test_predict_MLPGraphNetwork ------------ \n')
        # test with single block, non-shared params
        model = MLPGraphNetwork(n_blocks=1, share_params=False)
        self.forward_pass_helper_test(model)

        # test with two blocks, non-shared params
        model = MLPGraphNetwork(n_blocks=2, share_params=False)
        self.forward_pass_helper_test(model)

        # test with ten blocks, non-shared params
        model = MLPGraphNetwork(n_blocks=10, share_params=False)
        self.forward_pass_helper_test(model)

        # test with two blocks, shared params
        # TODO: add tests to check that params are actually shared
        model = MLPGraphNetwork(n_blocks=2, share_params=False)
        self.forward_pass_helper_test(model)

if __name__ == "__main__":
    # set up logging for unittest outputs
    log_path = f"tests/outputs/model_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    set_up_logging(log_path=log_path, log_level_str="INFO")

    with open(log_path, "a") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, verbosity=2)
