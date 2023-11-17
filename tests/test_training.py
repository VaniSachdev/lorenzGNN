from utils.jraph_data import get_lorenz_graph_tuples
from run_net import set_up_logging
import jax.numpy as jnp
import numpy as np
import unittest
import logging
import os
from datetime import datetime
import pdb

import jax.random

from utils.jraph_models import MLPBlock, MLPGraphNetwork
from utils.jraph_training import train_step, rollout_loss, evaluate_step, evaluate_model, train_and_evaluate
from tests.helpers import get_sample_data, state_setup_helper
from tests.mlp_sample_config import get_config

class TrainingTests(unittest.TestCase):

    # def setUp(self):
        # self.K = 36
        # self.F = 8
        # self.c = 10
        # self.b = 10
        # self.h = 1
        # self.seed = 42
        # self.output_steps = 2

    def test_rollout_loss(self):
        """ test that the non-batched rollout loss function works. """
        logging.info('\n ------------ test_rollout_loss ------------ \n')

        # alternatively, perhaps we could define some sample graphs where we know the expected rollout loss? 
        # actually that's complicated because we would have to manually set the params as well. nvm 

        sample_dataset, data_params = get_sample_data()

        sample_input_window = sample_dataset['train']['inputs'][0]
        sample_target_window = sample_dataset['train']['targets'][0]

        # set up state object, which helps us keep track of the model, params, and optimizer
        model = MLPBlock()
        state = state_setup_helper(model)

        # test single rollout 
        # call the function and make sure it doesn't crash 
        avg_loss, pred_nodes = rollout_loss(
            state=state, 
            n_rollout_steps=data_params['output_steps'],
            input_window_graphs=sample_input_window,
            target_window_graphs=sample_target_window,
            rngs=None,
        )
        
        # check that the loss is in a valid range (i.e. not negative)
        self.assertGreater(avg_loss, 0)

        # check the structure of the predicted nodes is valid 
        self.assertEqual(len(pred_nodes), data_params['output_steps']) # note that pred_nodes is a list containing a jax array for each rollout step 
        self.assertEqual(pred_nodes[0].shape, (data_params['K'], 2), f"pred_nodes shape is {pred_nodes[0].shape}")


    def test_train_step(self):
        """ test that the train_step() function works. """
        logging.info('\n ------------ test_train_step ------------ \n')
        sample_dataset, data_params = get_sample_data()

        sample_input_window = sample_dataset['train']['inputs'][0]
        sample_target_window = sample_dataset['train']['targets'][0]

        # set up model
        hidden_layer_features = {
            'edge': [16, 8], 
            'node': [32, 2], 
            'global': None}
        model = MLPBlock(layer_norm=False,
                         deterministic=False,
                         edge_features=hidden_layer_features['edge'],
                         node_features=hidden_layer_features['node'],
                         global_features=hidden_layer_features['global'])


        # set up state object, which helps us keep track of the model, params, and optimizer
        init_state = state_setup_helper(model=model)

        # test single training step 
        rng = jax.random.key(0)
        new_state, metrics_update, pred_nodes = train_step(
            state=init_state,
            n_rollout_steps=data_params['output_steps'],
            input_window_graphs=sample_input_window,
            target_window_graphs=sample_target_window,
            rngs={'dropout': rng}
        )

        # check that the state's step increased by 1 after training
        self.assertEqual(init_state.step, 0)
        self.assertEqual(new_state.step, 1)

        # check that metrics_update count was updated
        self.assertEqual(metrics_update.loss.count, 1)

        # check that the logged loss is valid
        self.assertGreater(float(metrics_update.loss.total), 0)

        # check that the number of predictions in the rollout is correct
        self.assertEqual(len(pred_nodes), data_params['output_steps']) # pred_nodes is a list of arrays 

        # check that the number of params is correct 
        self.assertEqual(
            init_state.params['params']['MLP_0']['Dense_0']['bias'].shape, 
            (hidden_layer_features['edge'][0], ))
        self.assertEqual(
            init_state.params['params']['MLP_0']['Dense_0']['kernel'].shape, 
            (6, hidden_layer_features['edge'][0])) 
        # the 6 input features for the edge_update mlp are from: 
        #   1 global feature 
        # + 2 sent attributes per edge (X1 and X2 from that node) 
        # + 2 received attributes per edge (X1 and X2 from the neighbor node) 
        # + 1 edge feature (indicating distance to feature node) 

        self.assertEqual(
            init_state.params['params']['MLP_0']['Dense_1']['bias'].shape, 
            (hidden_layer_features['edge'][1], ))
        self.assertEqual(
            init_state.params['params']['MLP_0']['Dense_1']['kernel'].shape, 
            (hidden_layer_features['edge'][0], 
             hidden_layer_features['edge'][1])) 
        
        self.assertEqual(
            init_state.params['params']['MLP_1']['Dense_0']['bias'].shape, 
            (hidden_layer_features['node'][0], ))
        self.assertEqual(
            init_state.params['params']['MLP_1']['Dense_0']['kernel'].shape, 
            (1+2+(2*hidden_layer_features['edge'][1]), 
             hidden_layer_features['node'][0]))
        # the 19 input features for the node_update mlp are from: 
        #   1 global feature 
        # + 2 node attributes (X1 and X2) 
        # + aggregated sender-node edge features from the edge_update mlp
        # + aggregated receiver-node edge features from the edge_update mlp

        self.assertEqual(
            init_state.params['params']['MLP_1']['Dense_1']['bias'].shape, 
            (hidden_layer_features['node'][1], ))
        self.assertEqual(
            init_state.params['params']['MLP_1']['Dense_1']['kernel'].shape, 
            (hidden_layer_features['node'][0], 
             hidden_layer_features['node'][1])) 


        # check that the params are different after training
        # check params for first MLP layer in update_edge_fn
        self.assertFalse(
            jnp.array_equal(
                init_state.params['params']['MLP_0']['Dense_0']['bias'],
                new_state.params['params']['MLP_0']['Dense_0']['bias'],
                ("init_state", init_state.params['params']['MLP_0']['Dense_0']['bias'],
                "new_state", new_state.params['params']['MLP_0']['Dense_0']['bias'])
            )
        )
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

    def test_evaluate_step(self):
        """ test that the evaluate_step() function works. """
        logging.info('\n ------------ test_evaluate_step ------------ \n')
        sample_dataset, data_params = get_sample_data()

        sample_input_window = sample_dataset['train']['inputs'][0]
        sample_target_window = sample_dataset['train']['targets'][0]

        # set up model
        hidden_layer_features = {
            'edge': [16, 8], 
            'node': [32, 2], 
            'global': None}
        model = MLPBlock(edge_features=hidden_layer_features['edge'],
                         node_features=hidden_layer_features['node'],
                         global_features=hidden_layer_features['global'])


        # set up state object, which helps us keep track of the model, params, and optimizer
        init_state = state_setup_helper(model=model)

        # test single eval step 
        # rng = jax.random.key(0)
        metrics_update, pred_nodes = evaluate_step(
            state=init_state,
            n_rollout_steps=data_params['output_steps'],
            input_window_graphs=sample_input_window,
            target_window_graphs=sample_target_window,
        )

        # check that metrics_update count was updated
        self.assertEqual(metrics_update.loss.count, 1)

        # check that the logged loss is valid
        self.assertGreater(float(metrics_update.loss.total), 0)

        # check that the number of predictions in the rollout is correct
        self.assertEqual(len(pred_nodes), data_params['output_steps']) # pred_nodes is a list of arrays 


    def test_evaluate_model(self):
        """ test that the evaluate_model() function works. """
        logging.info('\n ------------ test_evaluate_model ------------ \n')
        sample_dataset, data_params = get_sample_data()

        # set up model
        hidden_layer_features = {
            'edge': [16, 8], 
            'node': [32, 2], 
            'global': None}
        model = MLPBlock(edge_features=hidden_layer_features['edge'],
                         node_features=hidden_layer_features['node'],
                         global_features=hidden_layer_features['global'])

        # set up state object, which helps us keep track of the model, params, and optimizer
        init_state = state_setup_helper(model=model)

        # test evaluate_model 
        eval_metrics = evaluate_model(
            state=init_state,
            n_rollout_steps=data_params['output_steps'],
            datasets=sample_dataset,
            splits=['val', 'test']
        )

        # check that count was updated
        n_val_samples = int(data_params['n_samples']*data_params['val_pct'])
        n_test_samples = int(data_params['n_samples']*data_params['test_pct'])
        self.assertEqual(eval_metrics['val'].loss.count, n_val_samples)
        self.assertEqual(eval_metrics['test'].loss.count, n_test_samples)

        # check that the logged losses are valid
        self.assertGreater(float(eval_metrics['val'].loss.total), 0)
        self.assertGreater(float(eval_metrics['test'].loss.total), 0)

    def test_train_and_evaluate(self):
        """ test that the train_and_evaluate() function works. """
        logging.info('\n ------------ test_train_and_evaluate ------------ \n')
        mlp_config = get_config()
        workdir=f"tests/outputs/train_testing_dir_{datetime.now()}"

        # test that the function runs without crashing
        trained_state = train_and_evaluate(config=mlp_config, workdir=workdir)

        # check the state has the correct number of steps 
        num_train_steps = int(
            mlp_config.epochs * mlp_config.n_samples * mlp_config.train_pct
            )
        self.assertEqual(trained_state.step, num_train_steps)

        # check that the number of params is correct 
        self.assertEqual(
            trained_state.params['params']['MLP_0']['Dense_0']['bias'].shape, 
            (mlp_config.edge_features[0], ))
        self.assertEqual(
            trained_state.params['params']['MLP_0']['Dense_0']['kernel'].shape, 
            (6, mlp_config.edge_features[0])) 
        # the 6 input features for the edge_update mlp are from: 
        #   1 global feature 
        # + 2 sent attributes per edge (X1 and X2 from that node) 
        # + 2 received attributes per edge (X1 and X2 from the neighbor node) 
        # + 1 edge feature (indicating distance to feature node) 

        self.assertEqual(
            trained_state.params['params']['MLP_0']['Dense_1']['bias'].shape, 
            (mlp_config.edge_features[1], ))
        self.assertEqual(
            trained_state.params['params']['MLP_0']['Dense_1']['kernel'].shape, 
            (mlp_config.edge_features[0], 
             mlp_config.edge_features[1])) 
        
        self.assertEqual(
            trained_state.params['params']['MLP_1']['Dense_0']['bias'].shape, 
            (mlp_config.node_features[0], ))
        self.assertEqual(
            trained_state.params['params']['MLP_1']['Dense_0']['kernel'].shape, 
            (1+2+(2*mlp_config.edge_features[1]), 
             mlp_config.node_features[0]))
        # the 19 input features for the node_update mlp are from: 
        #   1 global feature 
        # + 2 node attributes (X1 and X2) 
        # + aggregated sender-node edge features from the edge_update mlp
        # + aggregated receiver-node edge features from the edge_update mlp

        self.assertEqual(
            trained_state.params['params']['MLP_1']['Dense_1']['bias'].shape, 
            (mlp_config.node_features[1], ))
        self.assertEqual(
            trained_state.params['params']['MLP_1']['Dense_1']['kernel'].shape, 
            (mlp_config.node_features[0], 
             mlp_config.node_features[1])) 


if __name__ == "__main__":
    # set up logging for unittest outputs
    log_path = f"tests/outputs/training_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    set_up_logging(log_path=log_path, log_level_str="INFO")

    with open(log_path, "a") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, verbosity=2)
