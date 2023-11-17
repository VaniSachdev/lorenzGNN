from utils.lorenz import get_window_indices, load_lorenz96_2coupled
from utils.jraph_data import get_lorenz_graph_tuples
from run_net import set_up_logging
import jax.numpy as jnp
import numpy as np
import unittest
import logging
import os
from datetime import datetime
import pdb


class DataTests(unittest.TestCase):

    def setUp(self):
        self.K = 36
        self.F = 8
        self.c = 10
        self.b = 10
        self.h = 1
        self.seed = 42

    def test_window_indices(self):
        """ test that the window indices are computed correctly. """
        logging.info('\n ------------ test_window_indices ------------ \n')

        # test 1
        n_samples_1 = 20
        init_buffer_samples_1 = 0
        timestep_duration_1 = 2
        input_steps_1 = 5
        output_delay_1 = 1
        output_steps_1 = 3
        sample_buffer_1 = 2

        x_windows_1, y_windows_1 = get_window_indices(
            n_samples=n_samples_1 + init_buffer_samples_1,
            timestep_duration=timestep_duration_1,
            input_steps=input_steps_1,
            output_delay=output_delay_1,
            output_steps=output_steps_1,
            sample_buffer=sample_buffer_1)

        # check that the number of windows is correct
        self.assertEqual(len(x_windows_1), n_samples_1 + init_buffer_samples_1)
        self.assertEqual(len(y_windows_1), n_samples_1 + init_buffer_samples_1)

        # spot-check that each window has the correct number of datapoints
        self.assertEqual(len(x_windows_1[0]), input_steps_1)
        self.assertEqual(len(y_windows_1[0]), output_steps_1)

        # spot-check that the first and last window have the correct start and end indices
        self.assertEqual(x_windows_1[0][0], 0)
        self.assertEqual(x_windows_1[0][-1], 8)
        self.assertEqual(y_windows_1[0][0], 12)
        self.assertEqual(y_windows_1[0][-1], 16)

        self.assertEqual(x_windows_1[-1][0], 418)
        self.assertEqual(x_windows_1[-1][-1], 426)
        self.assertEqual(y_windows_1[-1][0], 430)
        self.assertEqual(y_windows_1[-1][-1], 434)

        # test 2
        n_samples_2 = 100
        init_buffer_samples_2 = 100
        timestep_duration_2 = 3
        input_steps_2 = 6
        output_delay_2 = 0
        output_steps_2 = 4
        sample_buffer_2 = 2

        x_windows_2, y_windows_2 = get_window_indices(
            n_samples=n_samples_2 + init_buffer_samples_2,
            timestep_duration=timestep_duration_2,
            input_steps=input_steps_2,
            output_delay=output_delay_2,
            output_steps=output_steps_2,
            sample_buffer=sample_buffer_2)

        # check that the number of windows is correct
        self.assertEqual(len(x_windows_2), n_samples_2 + init_buffer_samples_2)
        self.assertEqual(len(y_windows_2), n_samples_2 + init_buffer_samples_2)

        # spot-check that each window has the correct number of datapoints
        self.assertEqual(len(x_windows_2[0]), input_steps_2)
        self.assertEqual(len(y_windows_2[0]), output_steps_2)

        # spot-check that the first and last window window have the correct start and end indices
        self.assertEqual(x_windows_2[0][0], 0)
        self.assertEqual(x_windows_2[0][-1], 15)
        self.assertEqual(y_windows_2[0][0], 18)
        self.assertEqual(y_windows_2[0][-1], 27)

        self.assertEqual(x_windows_2[-1][0], 7164)
        self.assertEqual(x_windows_2[-1][-1], 7179)
        self.assertEqual(y_windows_2[-1][0], 7182)
        self.assertEqual(y_windows_2[-1][-1], 7191)


    def test_graphtuple_datasets(self):
        """ test that the Lorenz data windows are sampling the correct values."""
        logging.info(
            '\n ------------ test_graphtuple_datasets ------------ \n')
        n_samples = 100
        input_steps = 6
        output_delay = 0
        output_steps = 4
        timestep_duration = 3
        sample_buffer = 2  # buffer between consequetive samples
        init_buffer_samples = 100  # buffer at the beginning of the dataset to allow for the system to settle
        time_resolution = 100
        data_path = "/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/data/test.npz"

        # generate desired dataset with train/val split and subsampled windows
        graph_tuple_dict = get_lorenz_graph_tuples(
            n_samples=n_samples,
            input_steps=input_steps,
            output_delay=output_delay,
            output_steps=output_steps,
            timestep_duration=timestep_duration,
            sample_buffer=sample_buffer,
            time_resolution=time_resolution,
            init_buffer_samples=init_buffer_samples,
            train_pct=0.7,
            val_pct=0.3,
            test_pct=0.0,
            K=self.K,
            F=self.F,
            c=self.c,
            b=self.b,
            h=self.h,
            seed=self.seed, 
            normalize=False) # TODO: test with normalized. perhaps just with plots)
        # graph_tuple_dict has the following format:
        # {
        # 'train': {
        #     'inputs': list of windows of graphtuples
        #     'targets': list of windows of graphtuples},
        # 'val': {
        #     'inputs': list of windows of graphtuples,
        #     'targets': list of windows of graphtuples},
        # 'test': {
        #     'inputs': list of windows of graphtuples,
        #     'targets': list of windows of graphtuples},
        # }
        
        # check sizes of datasets
        self.assertEqual(len(graph_tuple_dict['train']['inputs']),
                         int(n_samples * 0.7))
        self.assertEqual(len(graph_tuple_dict['train']['targets']),
                         int(n_samples * 0.7))
        self.assertEqual(len(graph_tuple_dict['val']['inputs']),
                         int(n_samples * 0.3))
        self.assertEqual(len(graph_tuple_dict['test']['inputs']), 0)

        # check number of data points in a window (i.e. a batched graphstuple)
        # check number of data points in a window
        self.assertEqual(len(graph_tuple_dict['train']['inputs'][0]),
                         input_steps)
        self.assertEqual(len(graph_tuple_dict['train']['targets'][0]),
                         output_steps)
        
        # check basic graph size attributes
        sample_graphtuple = graph_tuple_dict['train']['inputs'][0][0]
        self.assertEqual(sample_graphtuple.nodes.shape, (self.K, 2))
        self.assertEqual(sample_graphtuple.edges.shape, (self.K * 5, 1))
        self.assertEqual(sample_graphtuple.n_node[0], self.K)
        self.assertEqual(sample_graphtuple.n_edge[0], self.K * 5)

        # check data sampling (compare node 0 time series in subsampled vs raw dataset)
        # retrieve the "raw" Lorenz simulation data so we can compare with the sampled windows
        _, raw_data = load_lorenz96_2coupled(data_path)


        # check data sampling for first train input window, X1 variable
        data_sampled_first_train_input_window_X1 = np.vstack(
            [graph_tuple_dict['train']['inputs'][0][i].nodes[:, 0] for i in range(input_steps)])
        self.assertTrue(
            np.allclose(
                data_sampled_first_train_input_window_X1,
                raw_data[list(range(3600, 3615 + 1, 3)), :self.K]))
        # check data sampling for first train input window, X2 variable
        data_sampled_first_train_input_window_X2 = np.vstack(
            [graph_tuple_dict['train']['inputs'][0][i].nodes[:, 1] for i in range(input_steps)])
        self.assertTrue(
            np.allclose(data_sampled_first_train_input_window_X2,
                raw_data[list(range(3600, 3615 + 1, 3)), self.K:]))

        # check data sampling for last train target window, X1 variable
        data_sampled_last_train_target_window_X1 = np.vstack(
            [graph_tuple_dict['train']['targets'][-1][i].nodes[:, 0] for i in range(output_steps)])
        self.assertTrue(
            np.allclose(
                data_sampled_last_train_target_window_X1,
                raw_data[list(range(6102, 6111 + 1, 3)), :self.K]))

        # check data sampling for last val target window, X1 variable
        data_sampled_last_val_target_window_X1 = np.vstack(
            [graph_tuple_dict['val']['targets'][-1][i].nodes[:, 0] for i in range(output_steps)])
        self.assertTrue(
            np.allclose(
                data_sampled_last_val_target_window_X1,
                raw_data[list(range(7182, 7191 + 1, 3)), :self.K]))

    def test_normalization(self):
        """ test that the Lorenz graphstuples normalization preserves the correct data structures."""
        # TODO: test the normalization computations/values too 
        logging.info(
            '\n ------------ test_normalization ------------ \n')
        n_samples = 100
        input_steps = 6
        output_delay = 0
        output_steps = 4
        timestep_duration = 3
        sample_buffer = 2  # buffer between consequetive samples
        init_buffer_samples = 100  # buffer at the beginning of the dataset to allow for the system to settle
        time_resolution = 100
        # data_path = "/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/data/test.npz"

        # generate desired dataset with train/val split and subsampled windows
        graph_tuple_dict = get_lorenz_graph_tuples(
            n_samples=n_samples,
            input_steps=input_steps,
            output_delay=output_delay,
            output_steps=output_steps,
            timestep_duration=timestep_duration,
            sample_buffer=sample_buffer,
            time_resolution=time_resolution,
            init_buffer_samples=init_buffer_samples,
            train_pct=0.7,
            val_pct=0.3,
            test_pct=0.0,
            K=self.K,
            F=self.F,
            c=self.c,
            b=self.b,
            h=self.h,
            seed=self.seed, 
            normalize=True)
        # graph_tuple_dict has the following format:
        # {
        # 'train': {
        #     'inputs': list of windows of graphtuples
        #     'targets': list of windows of graphtuples},
        # 'val': {
        #     'inputs': list of windows of graphtuples,
        #     'targets': list of windows of graphtuples},
        # 'test': {
        #     'inputs': list of windows of graphtuples,
        #     'targets': list of windows of graphtuples},
        # }
        
        # check sizes of datasets
        self.assertEqual(len(graph_tuple_dict['train']['inputs']),
                         int(n_samples * 0.7))
        self.assertEqual(len(graph_tuple_dict['train']['targets']),
                         int(n_samples * 0.7))
        self.assertEqual(len(graph_tuple_dict['val']['inputs']),
                         int(n_samples * 0.3))
        self.assertEqual(len(graph_tuple_dict['test']['inputs']), 0)

        # check number of data points in a window (i.e. a batched graphstuple)
        # check number of data points in a window
        self.assertEqual(len(graph_tuple_dict['train']['inputs'][0]),
                         input_steps)
        self.assertEqual(len(graph_tuple_dict['train']['targets'][0]),
                         output_steps)
        
        # check basic graph size attributes
        sample_graphtuple = graph_tuple_dict['train']['inputs'][0][0]
        self.assertEqual(sample_graphtuple.nodes.shape, (self.K, 2))
        self.assertEqual(sample_graphtuple.edges.shape, (self.K * 5, 1))
        self.assertEqual(sample_graphtuple.n_node[0], self.K)
        self.assertEqual(sample_graphtuple.n_edge[0], self.K * 5)

if __name__ == "__main__":
    # set up logging for unittest outputs
    log_path = f"tests/outputs/data_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    set_up_logging(log_path=log_path, log_level_str="INFO")

    with open(log_path, "a") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, verbosity=2)
