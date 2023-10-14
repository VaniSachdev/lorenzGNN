from utils.lorenz import get_window_indices, lorenzToDF, lorenzDatasetWrapper, run_Lorenz96_2coupled
from run_net import set_up_logging
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

        # spot-check that the first and last window window have the correct start and end indices
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

    def test_lorenzToDF(self):
        logging.info('\n ------------ test_lorenzToDF ------------ \n')
        n_steps = 3591
        time_resolution = 100
        init_buffer_samples = 100

        df = lorenzToDF(K=self.K,
                        F=self.F,
                        c=self.c,
                        b=self.b,
                        h=self.h,
                        coupled=True,
                        n_steps=n_steps + init_buffer_samples + 1,
                        time_resolution=time_resolution,
                        seed=self.seed)
        logging.info(f"lorenz dataframe shape: {df.shape}")
        self.assertEqual(df.shape,
                         (n_steps + init_buffer_samples + 1, self.K * 2))

    def test_lorenz_sampling_window(self):
        """ test that the Lorenz data windows are sampled correctly in the lorenzDatasetWrapper for the X1X2_window prediction paradigm."""
        logging.info(
            '\n ------------ test_lorenz_sampling_window ------------ \n')
        n_samples = 100
        input_steps = 6
        output_delay = 0
        output_steps = 4
        timestep_duration = 3
        sample_buffer = 2  # buffer between consequetive samples
        init_buffer = 100  # buffer at the beginning of the dataset to allow for the system to settle
        time_resolution = 100

        # generate desired dataset with train/val split and subsampled windows
        dataset_subsampled = lorenzDatasetWrapper(
            predict_from="X1X2_window",
            n_samples=n_samples,
            # preprocessing=None, # ?????
            simple_adj=False,
            input_steps=input_steps,
            output_delay=output_delay,
            output_steps=output_steps,
            timestep_duration=timestep_duration,
            sample_buffer=sample_buffer,
            K=self.K,
            F=self.F,
            c=self.c,
            b=self.b,
            h=self.h,
            coupled=True,
            time_resolution=time_resolution,
            init_buffer_samples=init_buffer,
            return_buffer=True,
            seed=self.seed,
            override=True,
            train_pct=0.7,
            val_pct=0.3,
            test_pct=0.0)

        # check sizes of datasets
        self.assertEqual(dataset_subsampled.buffer.n_graphs, init_buffer)
        self.assertEqual(dataset_subsampled.train.n_graphs,
                         int(n_samples * 0.7))
        self.assertEqual(dataset_subsampled.val.n_graphs, int(n_samples * 0.3))
        self.assertIsNone(dataset_subsampled.test)

        # check basic graph size attributes
        self.assertEqual(dataset_subsampled.train[0].n_nodes, self.K)
        self.assertEqual(dataset_subsampled.train[0].n_node_features,
                         2 * input_steps)
        self.assertIsNone(dataset_subsampled.train[0].n_edge_features)
        self.assertEqual(dataset_subsampled.train[0].n_labels, 2 * output_steps)
        self.assertEqual(dataset_subsampled.train[0].x.shape,
                         (self.K, 2 * input_steps))
        self.assertEqual(dataset_subsampled.train[0].y.shape,
                         (self.K, 2 * output_steps))

        # check data sampling (check time values)
        # check time array for first train input window
        self.assertTrue(
            np.allclose(dataset_subsampled.train[0].t_X, (1 / time_resolution) *
                        np.arange(3600, 3615 + 1, timestep_duration)))
        # check time array for first train target window
        self.assertTrue(
            np.allclose(dataset_subsampled.train[0].t_Y, (1 / time_resolution) *
                        np.arange(3618, 3627 + 1, timestep_duration)))
        # check time array for last train input window
        self.assertTrue(
            np.allclose(dataset_subsampled.train[-1].t_X,
                        (1 / time_resolution) *
                        np.arange(6084, 6099 + 1, timestep_duration)))
        # check time array for last train target window
        self.assertTrue(
            np.allclose(dataset_subsampled.train[-1].t_Y,
                        (1 / time_resolution) *
                        np.arange(6102, 6111 + 1, timestep_duration)))

        # check data sampling (compare node 0 time series in subsampled vs raw dataset)

        # generate "raw" dataset of entire underlying Lorenz simulation so we can compare the sampled windows
        total_n_steps = timestep_duration * (
            (n_samples + init_buffer) *
            (input_steps + output_delay + output_steps + sample_buffer) -
            (sample_buffer + 1)) + 1  # = 7191 + 1 to account for 0-indexing

        dataset_raw_df = lorenzToDF(K=self.K,
                                    F=self.F,
                                    c=self.c,
                                    b=self.b,
                                    h=self.h,
                                    n_steps=total_n_steps,
                                    time_resolution=time_resolution,
                                    seed=self.seed)

        # check data sampling for first train input window, X1 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[0].x[:, :input_steps],
                dataset_raw_df.iloc[list(range(3600, 3615 + 1, 3)), :self.K].T))
        # check data sampling for first train input window, X2 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[0].x[:, input_steps:],
                dataset_raw_df.iloc[list(range(3600, 3615 + 1, 3)), self.K:].T))

        # check data sampling for first train target window, X1 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[0].y[:, :output_steps],
                dataset_raw_df.iloc[list(range(3618, 3627 + 1, 3)), :self.K].T))
        # check data sampling for first train target window, X2 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[0].y[:, output_steps:],
                dataset_raw_df.iloc[list(range(3618, 3627 + 1, 3)), self.K:].T))

        # check data sampling for last train input window, X1 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[-1].x[:, :input_steps],
                dataset_raw_df.iloc[list(range(6084, 6099 + 1, 3)), :self.K].T))
        # check data sampling for last train input window, X2 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[-1].x[:, input_steps:],
                dataset_raw_df.iloc[list(range(6084, 6099 + 1, 3)), self.K:].T))

        # check data sampling for last train target window, X1 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[-1].y[:, :output_steps],
                dataset_raw_df.iloc[list(range(6102, 6111 + 1, 3)), :self.K].T))
        # check data sampling for last train target window, X2 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.train[-1].y[:, output_steps:],
                dataset_raw_df.iloc[list(range(6102, 6111 + 1, 3)), self.K:].T))

        # check data sampling for last val target window, X1 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.val[-1].y[:, :output_steps],
                dataset_raw_df.iloc[list(range(7182, 7191 + 1, 3)), :self.K].T))
        # check data sampling for last val target window, X2 variable
        self.assertTrue(
            np.allclose(
                dataset_subsampled.val[-1].y[:, output_steps:],
                dataset_raw_df.iloc[list(range(7182, 7191 + 1, 3)), self.K:].T))

    def test_graphstuples(self):
        logging.info("\n ------------ test_graphstuples ------------ \n")


if __name__ == "__main__":
    # set up logging for unittest outputs
    log_path = f"tests/outputs/data_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    set_up_logging(log_path=log_path, log_level_str="INFO")

    with open(log_path, "a") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, verbosity=2)
