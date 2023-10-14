import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import *
from lorenz import lorenzToDF, lorenzDataset, lorenzDatasetWrapper, DEFAULT_TIME_RESOLUTION
from plotters import plot_data

logging.basicConfig(level=logging.DEBUG,
                    filename='unit_test_outputs/logs.txt',
                    filemode='w')


class DataTests(unittest.TestCase):
    """ test suite for data generation and manipulation. """

    def test_lorenzToDF(self):
        logging.info('\n test_lorenzToDF \n')
        K = 36
        n_days = 2
        time_resolution = DEFAULT_TIME_RESOLUTION
        init_buffer_samples = 100

        df = lorenzToDF(
            K=K,
            F=8,
            c=10,
            b=10,
            h=1,
            coupled=True,
            n_steps=None,  # 30 * 100,
            n_days=n_days,
            time_resolution=time_resolution,
            init_buffer_samples=init_buffer_samples,
            return_buffer=True,
            seed=42)
        self.assertEqual(
            df.shape, (n_days * time_resolution + init_buffer_samples, K * 2))

    def test_lorenzDatasetWrapper_X2single(self):
        logging.info(
            '\n test_lorenzDatasetWrapper where predict_from="X2" and there is a buffer \n'
        )
        K = 36
        n_samples = 200
        buffer = 50
        dataset = lorenzDatasetWrapper(predict_from="X2",
                                       n_samples=n_samples,
                                       K=K,
                                       F=8,
                                       c=10,
                                       b=10,
                                       h=1,
                                       coupled=True,
                                       time_resolution=DEFAULT_TIME_RESOLUTION,
                                       init_buffer_samples=buffer,
                                       return_buffer=True,
                                       seed=42,
                                       override=True,
                                       train_pct=1.0,
                                       val_pct=0.0,
                                       test_pct=0.0)

        # check sizes of datasets
        self.assertEqual(dataset.buffer.n_graphs, buffer)
        self.assertEqual(dataset.train.n_graphs, n_samples)
        self.assertIsNone(dataset.val)
        self.assertIsNone(dataset.test)

        # check graph size attributes
        self.assertEqual(dataset.train[0].n_nodes, K)
        self.assertEqual(dataset.train[0].n_node_features, 1)
        self.assertIsNone(dataset.train[0].n_edge_features)
        self.assertEqual(dataset.train[0].n_labels, 1)
        self.assertEqual(dataset.train[0].x.shape, (K, 1))
        self.assertEqual(dataset.train[0].y.shape, (K, 1))

        # check adjacency matrix
        node_5_connections = np.zeros(K)
        node_5_connections[2:7] = 1
        assert_array_equal(
            np.array(dataset.train.a.todense()[4])[0], node_5_connections)

        # check normalization
        X1_mean_of_nodes_per_timestep = []
        X1_var_of_nodes_per_timestep = []
        X2_mean_of_nodes_per_timestep = []
        X2_var_of_nodes_per_timestep = []

        # calculate means and vars from buffer and train data
        for g in np.concatenate([dataset.buffer, dataset.train]):
            X2_mean_of_nodes_per_timestep.append(g.x.mean())
            X2_var_of_nodes_per_timestep.append(g.x.var())
            X1_mean_of_nodes_per_timestep.append(g.y.mean())
            X1_var_of_nodes_per_timestep.append(g.y.var())

        X1_manual_mean = np.mean(X1_mean_of_nodes_per_timestep)
        X2_manual_mean = np.mean(X2_mean_of_nodes_per_timestep)

        X1_var_of_means = np.var(X1_mean_of_nodes_per_timestep)
        X2_var_of_means = np.var(X2_mean_of_nodes_per_timestep)

        # variance of set calculated from variance of partitions following
        # https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
        X1_manual_var = ((K - 1) / (K * n_samples - 1)) * (
            np.sum(X1_var_of_nodes_per_timestep) + X1_var_of_means * K *
            (n_samples - 1) / (K - 1))
        X2_manual_var = ((K - 1) / (K * n_samples - 1)) * (
            np.sum(X2_var_of_nodes_per_timestep) + X2_var_of_means * K *
            (n_samples - 1) / (K - 1))

        dataset.normalize()
        logging.debug('X1_manual_mean: {}'.format(X1_manual_mean))
        logging.debug('X1_default_mean: {}'.format(dataset.X1_mean))
        logging.debug('X2_manual_mean: {}'.format(X2_manual_mean))
        logging.debug('X2_default_mean: {}'.format(dataset.X2_mean))
        logging.debug('X1_manual_std: {}'.format(np.sqrt(X1_manual_var)))
        logging.debug('X1_default_std: {}'.format(dataset.X1_std))
        logging.debug('X2_manual_std: {}'.format(np.sqrt(X2_manual_var)))
        logging.debug('X2_default_std: {}'.format(dataset.X2_std))
        self.assertAlmostEqual(dataset.X1_mean, X1_manual_mean)
        self.assertAlmostEqual(dataset.X2_mean, X2_manual_mean)
        # TODO: rip wrong calculations?
        # self.assertAlmostEqual(dataset.X1_std, np.sqrt(X1_manual_var))
        # self.assertAlmostEqual(dataset.X2_std, np.sqrt(X2_manual_var))

        # plot data and save output
        fig, (ax0, ax1) = plot_data(dataset.buffer,
                                    dataset.train,
                                    test=None,
                                    node=0)
        plt.tight_layout()
        fig.savefig('unit_test_outputs/X2single_data')


class ModelTests(unittest.TestCase):
    """ test suite for model architecture. """

    def test_GCN2(self):
        logging.info('\n test_GCN2 \n')

    def test_GCN3(self):
        logging.info('\n test_GCN3 \n')


if __name__ == "__main__":
    # run ```python3 unit_test_db.py &> unit_test.txt``` to pipe outputs to
    # text file
    log_file = 'unit_test_outputs/results.txt'
    with open(log_file, "w") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner)
