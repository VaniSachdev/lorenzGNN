################################################################################
# This file contains helper functions for generating and handling data from    #
# the Lorenz 96 model.                                                         #
################################################################################

# imports
import os
import json 
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.data.dataset import Dataset
from spektral.datasets.utils import DATASET_FOLDER
from spektral.utils import gcn_filter
import jraph 
import jax.numpy as jnp

from datetime import datetime
import logging
import pdb

DEFAULT_TIME_RESOLUTION = 100
DATA_DIRECTORY_PATH = "/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/data/data_directory.json"


# wrapper for the lorenzDataset with train/test splitting and normalization bundled in
class lorenzDatasetWrapper():

    def __init__(
            self,
            predict_from,
            n_samples,
            input_steps,  
            output_delay,  
            output_steps,
            timestep_duration,
            sample_buffer,
            time_resolution,
            init_buffer_samples,
            return_buffer,
            train_pct,
            val_pct,
            test_pct,
            K=36,
            F=8,
            c=10,
            b=10,
            h=1,
            coupled=True,
            preprocessing=True, # TODO: check if this was only for GCN? we should need this for GraphNet right? 
            simple_adj=False,
            seed=42,
            override=False,
            ):
        """ Initialize a lorenzDatasetWrapper object. 
        
        Args:
            predict_from (str): prediction paradigm. Options are "X1X2_window", 
                in which the target X1 and X2 states are predicted from the 
                input X1 and X2 states; and "X2", in which the target X1 state 
                is predicted from the input X2 state.
            n_samples (int): number of samples (windows) to generate data for.
            input_steps (int): number of timesteps in each input window.
            output_delay (int): number of timesteps strictly between the end of 
                the input window and the start of the output window.
            output_steps (int): number of timesteps in each output window.
            timestep_duration (int): the sampling rate for data points from the 
                raw Lorenz simulation data, i.e. the number of raw simulation 
                data points between consecutive timestep samples, i.e. the 
                slicing step size. all data points are separated by this value.
            sample_buffer (int): number of timesteps strictly between the end 
                of one full sample and the start of the next sample.
            time_resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            init_buffer_samples (int): number of full samples (includes input 
                and output windows) to generate before the first training 
                sample to allow for the system to settle. can be saved to use 
                or ignore during normalization.      
            return_buffer (bool): whether or not to save the buffer samples in 
                a class attribute. if saved, they will contribute to the 
                normalization step. useful to save only if generating a tiny 
                training set and need more data points for normalization; 
                otherwise, recomment discarding. 
            train_pct (float): percentage of samples to use for training.
            val_pct (float): percentage of samples to use for validation.
            test_pct (float): percentage of samples to use for testing.
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            coupled (bool): whether to use the coupled 2-layer Lorenz96 model 
                or original 1-layer Lorenz96 model
            preprocessing (bool): whether or not to apply the gcn_filter 
                preprocessing step to add self-loops and normalize the 
                adjacency matrix. if False, the adjacency matrix will contain 
                self-loops but won't be normalized.  
            simple_adj (bool): ?????? TODO fix 
            NOTE: preprocessing and simple_adj are vestigial from the GCN and not needed for the jraph GraphNet models
            # TODO: rename preprocessing to something more descriptive
            seed (int): for reproducibility 
            override (bool): whether or not to regenerate data that was 
                already generated previously

        """
        logging.debug('initializing lorenzDatasetWrapper')
        assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001
        # use error term due to float errors

        # set up variables
        self.predict_from = predict_from
        self.n_samples = int(n_samples)
        self.preprocessing = preprocessing
        self.timestep_duration = timestep_duration
        self.K = K
        self.F = F
        self.c = c
        self.b = b
        self.h = h
        self.coupled = coupled
        self.time_resolution = int(time_resolution)
        self.init_buffer_samples = init_buffer_samples
        self.return_buffer = return_buffer
        self.seed = seed
        # TODO: rename buffer/sample_buffer to be more distinguishable

        if self.predict_from == "X2":
            raise NotImplementedError("X2 code is not up to date")
            self.input_steps = 1
            self.output_steps = 1
            self.output_delay = 0
            self.sample_buffer = 0
        else:
            assert self.predict_from == "X1X2_window"
            self.input_steps = int(input_steps)
            self.output_steps = int(output_steps)
            self.output_delay = int(output_delay)
            self.sample_buffer = int(sample_buffer)

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct

        # generate dataset
        dataset_raw = lorenzDataset(
            predict_from=self.predict_from,
            n_samples=self.n_samples + self.init_buffer_samples,
            preprocessing=self.preprocessing,
            simple_adj=simple_adj,
            input_steps=self.input_steps,
            output_delay=self.output_delay,
            output_steps=self.output_steps,
            timestep_duration=self.timestep_duration,
            sample_buffer=self.sample_buffer,
            K=self.K,
            F=self.F,
            c=self.c,
            b=self.b,
            h=self.h,
            coupled=self.coupled,
            time_resolution=self.time_resolution,
            seed=self.seed,
            override=override)
        # split dataset
        if init_buffer_samples > 0 and return_buffer:
            self.buffer = dataset_raw[:init_buffer_samples]
        else:
            self.buffer = None

        dataset = dataset_raw[init_buffer_samples:]

        train_bound = round(train_pct * dataset.n_graphs)
        val_bound = round((train_pct + val_pct) * dataset.n_graphs)

        self.train = dataset[:train_bound] # check dataset dimensions if multiple inputs/outputs 
        self.val = None if train_pct == 1 else dataset[train_bound:val_bound]
        self.test = None if train_pct + val_pct == 1 else dataset[val_bound:]

    def normalize(self):
        """ normalize dataset using training data distribution, and buffer data 
            distribution if available. 

            (replaced existing train, val, test with normalized versions)
        """
        norm_input = self.train if self.buffer is None else self.buffer + self.train
        # including the buffer data helps to stabilize the mean and std when the train data is very small
        # but theoretically should drop once we use larger datasets because the buffer is throwaway data (to give the Lorenz model time to settle)
        # TODO: add flag to keep/drop buffer data in normalization

        self.X1_mean, self.X1_std, self.X2_mean, self.X2_std = norm_input.get_mean_std()
        if self.buffer is not None:
            self.buffer.normalize(self.X1_mean, self.X1_std, self.X2_mean,
                                  self.X2_std)
        self.train.normalize(self.X1_mean, self.X1_std, self.X2_mean,
                             self.X2_std)
        if self.val is not None:
            self.val.normalize(self.X1_mean, self.X1_std, self.X2_mean,
                               self.X2_std)
        if self.test is not None:
            self.test.normalize(self.X1_mean, self.X1_std, self.X2_mean,
                                self.X2_std)


# create dataset class for lorenz96 model
class lorenzDataset(Dataset):
    """ A dataset containing windows of data from a Lorenz96 time series. """

    def __init__(
            self,
            predict_from,
            n_samples,
            input_steps, 
            output_delay,  
            output_steps,
            timestep_duration,
            sample_buffer,
            time_resolution,
            K=36,
            F=8,
            c=10,
            b=10,
            h=1,
            coupled=True,
            preprocessing=True,
            simple_adj=False,
            seed=42,
            override=False,
            **kwargs):
        """ 
        Args:
            predict_from (str): prediction paradigm. Options are "X1X2_window", 
                in which the target X1 and X2 states are predicted from the 
                input X1 and X2 states; and "X2", in which the target X1 state 
                is predicted from the input X2 state.
            n_samples (int): number of samples (windows) to generate data for.
            input_steps (int): number of timesteps in each input window.
            output_delay (int): number of timesteps strictly between the end of 
                the input window and the start of the output window.
            output_steps (int): number of timesteps in each output window.
            timestep_duration (int): the sampling rate for data points from the 
                raw Lorenz simulation data, i.e. the number of raw simulation 
                data points between consecutive timestep samples, i.e. the 
                slicing step size. all data points are separated by this value.
            sample_buffer (int): number of timesteps strictly between the end 
                of one full sample and the start of the next sample.
            time_resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            coupled (bool): whether to use the coupled 2-layer Lorenz96 model 
                or original 1-layer Lorenz96 model
            preprocessing (bool): whether or not to apply the gcn_filter 
                preprocessing step to add self-loops and normalize the 
                adjacency matrix. if False, the adjacency matrix will contain 
                self-loops but won't be normalized.  
            simple_adj (bool): ?????? TODO fix 
            seed (int): for reproducibility 
            override (bool): whether or not to regenerate data that was 
                already generated previously
            **kwargs: additional arguments to pass to the Dataset superclass        
        """
        self.preprocessing = preprocessing
        self.simple_adj = simple_adj # TODO: add to docstring 
        self.predict_from = predict_from
        self.n_samples = int(n_samples)
        self.timestep_duration = timestep_duration
        self.K = K
        self.F = F
        self.c = c
        self.b = b
        self.h = h
        self.coupled = coupled
        self.time_resolution = int(time_resolution)
        self.seed = seed
        self.a = None  # self.a is set to None anyway in super __init__ so we have to define it in the read() function

        if self.predict_from == "X2":
            self.input_steps = 1
            self.output_steps = 1
            self.output_delay = 0
            self.sample_buffer = 0
        else:
            self.input_steps = int(input_steps)
            self.output_steps = int(output_steps)
            self.output_delay = int(output_delay)
            self.sample_buffer = int(sample_buffer)

        if override and os.path.exists(self.path):
            os.remove(self.path)

        super().__init__(**kwargs)

    @property
    def path(self):
        """ define the file path where data will be stored/extracted. """
        drive_base_path = '/content/drive/My Drive/_research ML AQ/lorenz 96 gnn/lorenz_data'  # NOTE: obviously this must change depending on your own computer's file system
        filename = f"{self.predict_from}_{self.n_samples}_{self.input_steps}_{self.output_steps}_{self.output_delay}_{self.timestep_duration}_{self.sample_buffer}_{self.K}_{self.F}_{self.c}_{self.b}_{self.h}_{self.coupled}_{self.time_resolution}_{self.seed}.npz"

        # check if we're in colab or not because the file paths will be different
        if not os.path.exists(drive_base_path):
            # either running locally or on colab without designated folder/matching file system structure
            if os.getcwd().startswith('/content/drive'):  # running on colab
                logging.info(
                    'using default root path to directory for storing generated lorenz datasets'
                )
            subpath = os.path.join('Lorenz', filename)
            path = os.path.join(DATASET_FOLDER, subpath)
        else:  # running on colab with custom directory for storing datasets
            logging.info('storing generated datasets in designated folder')
            path = os.path.join(drive_base_path, filename)
        return path

    def get_config(self):
        """ Retrieve a dictionary containing the configuration of the dataset."""
        return dict(predict_from=self.predict_from,
                    n_samples=self.n_samples,
                    input_steps=self.input_steps,
                    output_steps=self.output_steps,
                    output_delay=self.output_delay,
                    sample_buffer=self.sample_buffer,
                    time_resolution=self.time_resolution,
                    timestep_duration=self.timestep_duration,
                    K=self.K,
                    F=self.F,
                    c=self.c,
                    b=self.b,
                    h=self.h,
                    coupled=self.coupled,
                    preprocessing=self.preprocessing,
                    simple_adj=self.simple_adj,
                    seed=self.seed)

    def read(self):
        """ Reads stored dataset and returns a list of Graph objects. 

            This a function that all subclasses of Dataset must implement, and is automatically called in the superclass __init__.

            Assumes that the dataset file path already exists. (this is handled in super().__init__)
        """
        logging.info('reading Lorenz data from stored file')

        # create sparse adjacency matrix
        self.a = self.compute_adjacency_matrix()
        logging.debug('done computing adj')

        # read data from computer
        data = np.load(self.path, allow_pickle=True)
        # data = np.load(os.path.join(self.path, "data.npz"))

        if self.predict_from == 'X1X2_window':
            X = data['X']
            Y = data['Y']
            t_X = data['t_X']
            t_Y = data['t_Y']

            # convert to Graph structure
            return [
                Graph(x=X[i], y=Y[i], t_X=t_X[i], t_Y=t_Y[i])
                for i in range(self.n_samples)
            ]

        if self.predict_from == 'X2':
            X = data['X']
            Y = data['Y']
            t = data['t']

            # convert to Graph structure
            return [
                Graph(x=X[i], y=Y[i], t=t[i]) for i in range(self.n_samples)
            ]

    def download(self):
        """ Generate and store Lorenz data. 
        
            This a function that all subclasses of Dataset must implement, and is automatically called in the superclass __init__.
        """
        # Create the directory
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        filename = os.path.splitext(self.path)[0]

        logging.info('generating new Lorenz data and saving to file')
        # generate a sequence of windows to determine how our samples will be
        # spaced out
        # x_windows is a list of <n_samples> tuples; each element is a tuple
        # containing the first (inclusive) and last (non-inclusive) indices of
        # the input data sample
        # y_windows is the same but for the output data sample
        if self.predict_from == 'X1X2_window':
            X, Y, t_X, t_Y = self.generate_window_data()
            # Write the data to file
            np.savez(filename, X=X, Y=Y, t_X=t_X, t_Y=t_Y)

        elif self.predict_from == 'X2':
            X, Y, t = self.generate_paired_data()
            # Write the data to file
            np.savez(filename, X=X, Y=Y, t=t)

        elif self.predict_from == 'X2_window':
            raise NotImplementedError

        else:
            raise ValueError('invalid input for prediction_from argument')

    def generate_window_data(self):
        """ Generate data samples containing input and target windows for the 
            X1X2_window prediction paradigm. 
        
            Returns:
                X: list of input data windows, each which is a 2d np array of shape (K, input_steps)
                Y: list of target data windows, each which is a 2d np array of shape (K, output_steps)
                t_X: list of input data window timestamps, each which is a 1d np array of size (input_steps)
                t_Y: list of target data window timestamps, each which is a 1d np array of size (output_steps)
        """
        logging.info('generating window data')
        # get indices for the datapoints in each sample
        # x/y_windows are lists of lists; each sublist contains the indices for the datapoints in a single sample
        # x_windows contains the indices for input graphs and y_windows contains the indices for target graphs
        x_windows, y_windows = get_window_indices(
            n_samples=self.n_samples, 
            timestep_duration=self.timestep_duration, input_steps=self.input_steps, output_delay=self.output_delay, output_steps=self.output_steps, sample_buffer=self.sample_buffer)

        # generate some data

        # compute the number of total "raw" steps in the simulation we need
        if len(y_windows[-1]) > 0:
            n_steps = y_windows[-1][-1] + 1 
            # i.e. the last index in the last target data point
            # add one to account for the zero-indexing
        else:
            # if there are no target data points, then we need to use the last index in the last input data point
            # add one to account for the zero-indexing
            n_steps = x_windows[-1][-1] + 1


        logging.debug('total steps: {}'.format(n_steps))
        if self.coupled:
            lorenz_buffered_df = lorenzToDF(
                K=self.K,
                F=self.F,
                c=self.c,
                b=self.b,
                h=self.h,
                coupled=self.coupled,
                n_steps=n_steps,
                time_resolution=self.time_resolution,
                seed=self.seed)
        else:
            raise NotImplementedError

        X = []
        Y = []
        t_X = []
        t_Y = []

        for x_window, y_window in zip(x_windows, y_windows):

            # index and reshape dfs
            # originally, the dataframe has time indexing the rows and node features on the columns; we want to reshape this so that we have rows indexed by node and columns to contain the node feature at every time step

            x_df = lorenz_buffered_df.iloc[x_window]
            t_x = x_df.index
            x_df = x_df.T
            x_df = pd.concat([
                x_df.iloc[:self.K].reset_index(drop=True),
                x_df.iloc[self.K:].reset_index(drop=True)
            ], axis=1)

            y_df = lorenz_buffered_df.iloc[y_window]
            t_y = y_df.index
            y_df = y_df.T
            y_df = pd.concat([
                y_df.iloc[:self.K].reset_index(drop=True),
                y_df.iloc[self.K:].reset_index(drop=True)
            ], axis=1)

            # rename columns
            x_df.columns = ['X1_{}'.format(t)
                            for t in t_x] + ['X2_{}'.format(t) for t in t_x]
            y_df.columns = ['X1_{}'.format(t) for t in t_y] + ['X2_{}'.format(t) for t in t_y]

            # note that spektral graphs can't handle dataframes;
            # data must be in nparrays
            X.append(x_df.to_numpy())
            Y.append(y_df.to_numpy())
            t_X.append(t_x.to_numpy())
            t_Y.append(t_y.to_numpy())

        return X, Y, t_X, t_Y

    def generate_paired_data(self):
        # NOTE: WARNING, NOT MAINTAINED; DO NOT USE
        raise NotImplementedError('not maintained; do not use')
        logging.info('generating paired data')
        if self.coupled:
            lorenz_buffered_df = lorenzToDF(
                K=self.K,
                F=self.F,
                c=self.c,
                b=self.b,
                h=self.h,
                coupled=self.coupled,
                n_steps=self.
                n_samples,  # since 1 sample/window => n_samples total steps
                time_resolution=self.time_resolution,
                seed=self.seed)

        else:
            raise NotImplementedError

        X = lorenz_buffered_df.iloc[:,
                                    self.K:].to_numpy()  # i.e. the X2 variable
        Y = lorenz_buffered_df.iloc[:, :self.K].to_numpy(
        )  # i.e. the X1 variable we want to predict
        t = lorenz_buffered_df.index.to_numpy()

        X = X[:, :, np.newaxis]  # reshape to avoid warning from spektral
        Y = Y[:, :, np.newaxis]

        return X, Y, t

    def compute_adjacency_matrix(self):
        """ Generate adjacency matrix for the nodes on the circle graph. """
        if self.simple_adj:
            target_nodes = np.mod(np.arange(self.K) + 1, self.K)
            src_nodes = np.arange(self.K)
        else:
            target_nodes = np.mod(
                np.concatenate([
                    (np.arange(self.K) - 1),
                    (np.arange(self.K) - 2),
                    (np.arange(self.K) + 1),
                    (np.arange(self.K) + 2),
                ]), self.K)
            if not self.preprocessing:
                # if we aren't using the preprocessing transform that adds the identity matrix, we need to add self-loops outselves here
                target_nodes = np.concatenate([target_nodes, np.arange(self.K)])
                src_nodes = np.concatenate([np.arange(self.K)] * 5)
            else:
                src_nodes = np.concatenate([np.arange(self.K)] * 4)
        weights = np.ones(shape=len(src_nodes))
        a = coo_matrix((weights, (src_nodes, target_nodes)),
                       shape=(self.K, self.K))
        if self.preprocessing:
            a = gcn_filter(a)
        return a

    def get_mean_std(self):
        """ Calculates the mean and stdev for 1) all X1 variables (includes 
            both feature and target data), and 2) for all X2 variables
        
            Returns:
                4-tuple: X1_mean, X1_std, X2_mean, X2_std
        """
        # get one mean/stdev for all X1 variables (includes the x and y data), and one mean/stdev for all X2 variables
        # TODO: normalze should only be on train input data

        start = datetime.now()
        all_x = np.concatenate([g.x for g in self])
        all_y = np.concatenate([g.y for g in self])
        finish_concat = datetime.now()

        # TODO: verify if we actually want the X1 mean to be affected by output (X1 prediction) values?
        # (i guess when its X2_single mode, we still have to normalize the output data according to the training output.. hm ok)
        if self.predict_from == "X1X2_window":
            all_X1 = np.concatenate([all_x[:, :self.input_steps], all_y],
                                    axis=1)
            all_X2 = all_x[:, self.input_steps:]
        elif self.predict_from == "X2":
            all_X1 = all_y
            all_X2 = all_x
        else:
            raise NotImplementedError

        X1_mean = all_X1.mean()
        X1_std = all_X1.std()

        X2_mean = all_X2.mean()
        X2_std = all_X2.std()
        finish_extract = datetime.now()

        logging.debug('time to concat: {}'.format(finish_concat - start))
        logging.debug('time to get std&mean: {}'.format(finish_extract -
                                                        finish_concat))

        return X1_mean, X1_std, X2_mean, X2_std

    def normalize(self, X1_mean, X1_std, X2_mean, X2_std):
        """ Normalize the data in-place, given desired means and standard 
            deviations for X1 and X2. 
        """
        if self.predict_from == "X1X2_window":
            for g in self:
                # separate X1 and X2 in g.x. recall that g.x has shape
                # (K, 2 * input_steps)
                X1 = g.x[:, :self.input_steps]
                X2 = g.x[:, self.input_steps:]

                X1_norm = (X1 - X1_mean) / X1_std
                X2_norm = (X2 - X2_mean) / X2_std

                g.x = np.concatenate([X1_norm, X2_norm], axis=1)
                g.y = (g.y - X1_mean) / X1_std
                # (the target only contains the X1 variable)
        elif self.predict_from == "X2":
            for g in self:
                g.x = (g.x - X2_mean) / X2_std
                g.y = (g.y - X1_mean) / X1_std
        else:
            raise NotImplementedError

    def plot(self,
             node=0,
             fig=None,
             ax0=None,
             ax1=None,
             data_type='',
             color='darkcyan',
             alpha=1):
        """
            Args: 
                node (int): the node for which time-series data will be plotted
                ax0, ax1: (optional) matplotlib axes objects. If None, a new fig with 2 subplots will be created; otherwise, data will be plotted on the existing axes. 
                data_type (str): (optional) e.g. 'train', 'val', or 'test'
                color (str): (optional) color of the lines

            Returns: fig and axes
        """
        if fig is None or ax0 is None or ax1 is None:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 8))
            fig.suptitle("sampled time series after reshaping", size=28)
            ax0.set_title(
                "X1 (i.e. atmospheric variable) for node {}".format(node),
                size=20)
            ax1.set_title("X2 (i.e. oceanic variable) for node {}".format(node),
                          size=20)
            plt.xlabel('time (days)', size=16)

        if self.predict_from == "X1X2_window":
            for g in self:
                # plot X1
                ax0.scatter(g.t_X,
                            g.x[node][:self.input_steps],
                            label=data_type + ' inputs',
                            c=color,
                            alpha=alpha)
                # plot X2
                ax1.scatter(g.t_X,
                            g.x[node][self.input_steps:],
                            label=data_type + ' inputs',
                            c=color,
                            alpha=alpha)
                ax0.scatter(g.t_Y,
                            g.y[node][:self.output_steps],
                            label=data_type + ' labels',
                            s=30,
                            c=color,
                            alpha=alpha)
        elif self.predict_from == "X2":
            for g in self:
                ax1.scatter(g.t,
                            g.x[node],
                            label=data_type + ' inputs',
                            c=color,
                            alpha=alpha)
                ax0.scatter(g.t,
                            g.y[node],
                            label=data_type + ' labels',
                            s=30,
                            c=color,
                            alpha=alpha)
        else:
            raise NotImplementedError

        return fig, (ax0, ax1)

def get_window_indices(n_samples, timestep_duration, input_steps, output_delay, 
                       output_steps, sample_buffer):
    """ Compute indices for the datapoints in each sample. 
    
        Assumes a prediction paradigm of X1X2_window (i.e. we are predicting a 
        rollout into the future, given a window of past data).
    
        Args:
            n_samples (int): number of samples (windows) to generate data for.
            timestep_duration (int): the sampling rate for data points from the 
                raw Lorenz simulation data, i.e. the number of raw simulation 
                data points between consecutive timestep samples, i.e. the 
                slicing step size. all data points are separated by this value.
            input_steps (int): number of timesteps in each input window.
            output_delay (int): number of timesteps strictly between the end of 
                the input window and the start of the output window.
            output_steps (int): number of timesteps in each output window.
            sample_buffer (int): number of timesteps strictly between the end 
                of one full sample and the start of the next sample.

        Returns:
            x_windows (list of lists): each sublist contains the indices for 
                the datapoints for the inputs of a single sample
            y_windows (list of lists): each sublist contains the indices for 
                the datapoints for the targets of a single sample
            
    """
    x_windows = []
    y_windows = []

    for i in range(n_samples):
        input_start = i * timestep_duration * (
            input_steps + output_delay + output_steps + sample_buffer)
        input_end = input_start + timestep_duration * (
            input_steps - 1)
        
        output_start = input_end + timestep_duration * (output_delay + 1)
        output_end = output_start + timestep_duration * (output_steps - 1)

        x_windows.append(np.arange(input_start, input_end+1, timestep_duration, dtype=int))
        y_windows.append(np.arange(output_start, output_end+1, timestep_duration, dtype=int))

    return x_windows, y_windows

def lorenzToDF(
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        coupled=True,
        n_steps=100,
        time_resolution=100,
        seed=42):
    """ Generate a dataframe of data from the lorenz model. 

        Args: 
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            coupled (bool): whether to use the coupled 2-layer Lorenz96 model 
                or original 1-layer Lorenz96 model
            n_steps (int): number of raw timesteps for which to run the ODE 
                integration of the model (NOTE: this is distinct from the 
                number of steps in the LorenzDataset/Wrapper object, which is 
                sampled from this raw data. n_steps in this function would need 
                to be computed given the specific parameters passed to the 
                LorenzDataset/Wrapper object.)
            time_resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            seed (int): for reproducibility 

        Returns:
            df (pandas DataFrame): a dataframe with shape (n_steps, 2*K). The 
            rows are indexed by time; the first K columns contain the X1 data 
            for each respective node, and the latter K columns contain the X2 
            data for each respective node.
    """
    if coupled:
        t, X, _, _, _ = run_lorenz96_2coupled(
            K=K,
            F=F,
            c=c,
            b=b,
            h=h,
            n_steps=n_steps,
            resolution=time_resolution,
            seed=seed)
    else:
        raise NotImplementedError

    df = pd.DataFrame(X,
                      columns=['X1_{}'.format(i) for i in range(K)] +
                      ['X2_{}'.format(i) for i in range(K)],
                      index=t)
    df.index.name = 'time'
    return df


#####################
# OG Lorenz96 model #
#####################
def lorenz96(X, t, K, F):
    """ Functions defining a single update step in the Lorenz96 system.
    
        Copied from Prof. Kavassalis.

        Args: 
            X (float array, size K): array of X state values
            t: 
            K (int): number of points on the circumference
            F (float): forcing constant
    """
    #K-component Lorenz 96 model
    dX_dt = np.zeros(K)
    # boundary conditions
    # (define the wrapping)
    dX_dt[0] = (X[1] - X[K - 2]) * X[K - 1] - X[0] + F
    dX_dt[1] = (X[2] - X[K - 1]) * X[0] - X[1] + F
    dX_dt[K - 1] = (X[0] - X[K - 3]) * X[K - 2] - X[K - 1] + F
    # Then the general case
    for i in range(2, K - 1):
        dX_dt[i] = (X[i + 1] - X[i - 2]) * X[i - 1] - X[i] + F
    # Return the state derivatives
    return dX_dt


def run_lorenz96(K=36, F=8, number_of_days=30, nudge=True):
    """ (from Prof. Kavassalis) """
    X0 = F * np.ones(K)  # Initial state (equilibrium)
    if nudge == True:
        X0[random.randint(1, K) -
           1] = X0[random.randint(1, K) - 1] + random.uniform(
               0, .01)  # adds our perturbation
    t = np.arange(0.0, number_of_days,
                  0.01)  # creates the time points we want to see solutiosn for

    logging.info('starting integration')
    X = odeint(lorenz96, X0, t, args=(K, F),
               ixpr=True)  #solves the system of ordinary differential equations

    return t, X, F, K, number_of_days


##################################
# 2-layer coupled Lorenz96 model #
##################################


def lorenz96_2coupled(X, t, K, F, c, b, h):
    """ Functions defining a single update step in the coupled 2-layer Lorenz96 
        system.

        Copied from Prof. Kavassalis.

        Args: 
            X (float array, size 2*K): array of current X1 and X2 state values
            t: 
            K (int): number of points on the circumference
            F (float): forcing constant
            c (float): time-scale ratio ??
            b (float): spatial-scale ratio ??
            h (float): coupling parameter ??

        Returns:
            dX_dt (float array, size 2*K): array of the derivatives of the X1 and X2 state values at the given instant in time. 
        """
    dX_dt = np.zeros(K * 2)
    
    ######## first ##########
    # boundary conditions
    dX_dt[0] = (X[1] - X[K - 2]) * X[K - 1] - X[0] - (h * c / b) * X[K] + F
    dX_dt[1] = (X[2] - X[K - 1]) * X[0] - X[1] - (h * c / b) * X[K + 1] + F
    dX_dt[K -
          1] = (X[0] - X[K - 3]) * X[K - 2] - X[K -
                                                1] - (h * c / b) * X[K - 1] + F
    ######## second next #############
    # boundary conditions
    dX_dt[K + 0] = -c * b * (X[K + 2] - X[K + K - 1]) * X[K + 1] - c * X[K] + (
        h * c / b) * X[0]
    dX_dt[K + K -
          1] = -c * b * (X[K + 1] - X[K + K - 2]) * X[K] - c * X[K + K - 1] + (
              h * c / b) * X[K - 1]
    dX_dt[K + K - 2] = -c * b * (X[K + 2] - X[K + K - 3]) * X[
        K + K - 1] - c * X[K + K - 2] + (h * c / b) * X[K - 2]

    ######### first first ######################
    # Then the general case
    for i in range(2, K - 1):
        dX_dt[i] = (X[i + 1] - X[i - 2]) * X[i - 1] - X[i] - (h * c /
                                                              b) * X[i + K] + F
    # Return the state derivatives
    ######## second next #############################
    for i in range(K + 1, K + K - 2):
        dX_dt[i] = -c * b * (X[i + 2] - X[i - 1]) * X[i + 1] - c * X[i] + (
            h * c / b) * X[i - K]

    return dX_dt


def run_lorenz96_2coupled(
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        n_steps=300,
        resolution=DEFAULT_TIME_RESOLUTION,  # 100
        seed=42):
    """ Run ODE integration over the coupled 2-layer Lorenz96 model.
    
        Modified from Prof. Kavassalis.
    
        Args:
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            n_steps (int): number of raw timesteps for which to run the ODE 
                integration of the model (NOTE: this is distinct from the 
                number of steps in the LorenzDataset/Wrapper object, which is 
                sampled from this raw data. n_steps in this function would need 
                to be computed given the specific parameters passed to the 
                LorenzDataset/Wrapper object.)
            resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            seed (int): for reproducibility 

        Returns:
            t (float array): array of time points
            X (float array): array of state values at each time point
            F (float): forcing constant
            K (int): number of points on the circumference
            n_steps (int): number of time steps
    """
    random.seed(seed)

    # Initial state (equilibrium)
    X0 = np.concatenate((F * np.ones(K), (h * c / b) * np.ones(K)))

    # Perturbation
    X0[random.randint(0, K) -
       1] = X0[random.randint(0, K) - 1] + random.uniform(0, .01)

    simulation_duration = n_steps / resolution # number of time units
    t = np.arange(
        0.0, simulation_duration, 1 / resolution) # indices of all time steps

    logging.info('starting integration')
    X = odeint(lorenz96_2coupled,
               X0,
               t,
               args=(K, F, c, b, h),
               ixpr=True)

    return t, X, F, K, n_steps

def run_download_lorenz96_2coupled(
        fname, 
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        n_steps=300,
        resolution=DEFAULT_TIME_RESOLUTION,  # 100
        seed=42):
    """ Run ODE integration over the coupled 2-layer Lorenz96 model and save 
        the data to a .npz file. 
    
        Args: 
            fname (str): path and name of file to which the data will be saved.
            K (int): number of nodes on the circumference of the Lorenz96 model
            F (float): Lorenz96 forcing constant. (K=36 and F=8 corresponds to 
                an error-doubling time of 2.1 days, similar to the real 
                atmosphere)
            c (float): Lorenz96 time-scale ratio ?
            b (float): Lorenz96 spatial-scale ratio ?
            h (float): Lorenz96 coupling parameter ?
            n_steps (int): number of raw timesteps for which to run the ODE 
                integration of the model (NOTE: this is distinct from the 
                number of steps in the LorenzDataset/Wrapper object, which is 
                sampled from this raw data. n_steps in this function would need 
                to be computed given the specific parameters passed to the 
                LorenzDataset/Wrapper object.)
            resolution (int): the inverse of the delta t used in the 
                Lorenz ODE integration (∆t = 1/time_resolution); the number of 
                raw data points generated per time unit, equivalent to the 
                number of data points generated per 5 days in the simulation.
            seed (int): for reproducibility 

        Output:
            an .npz file containing t, the array of time points, and X, the array of state values at each time point. The parameters for the simulation run will also be saved to a data directory for reference.

            The data can be accessed similar to a dictionary, as follows: 
                data = np.load(fname, allow_pickle=True)
                t = data['t'] # array of time points
                X = data['X'] # array of state values at each time point, shape (?, ?)
    """
    # generate data
    t, X, _, _, _ = run_lorenz96_2coupled(K=K, F=F, c=c, b=b, h=h, n_steps=n_steps, resolution=resolution, seed=seed)

    # save data 
    np.savez(fname, t=t, X=X)

    # save params to the data directory
    # this is a json that contains the parameters and the file name, so that they can be logged and looked up 
    # the json consists of a list of dictionaries containing the params and file name

    # setup directory for data, if it doesn't exist 
    if not os.path.exists(DATA_DIRECTORY_PATH):
        os.makedirs(os.path.dirname(DATA_DIRECTORY_PATH), exist_ok=True)
        data_directory = []
    else: 
        # load json 
        with open(DATA_DIRECTORY_PATH, "r") as f:
            data_directory = json.load(f)

    # log the information for this data simulation 
    params = {
        "fname": fname, 
        "K": K,
        "F": F,
        "c": c,
        "b": b,
        "h": h,
        "n_steps": n_steps,
        "resolution": resolution,
        "seed": seed,
    }
    data_directory.append(params)

    # save data directory 
    with open(DATA_DIRECTORY_PATH, "w") as f:
        json.dump(data_directory, f, indent=4)



def load_lorenz96_2coupled(fname):
    """ Retrieves the lorenz96 data that was saved to a .npz file. 
    
        Args: 
            fname (str): path to npz file.

        Returns:
            t (float array): array of time points
            X (float array): array of state values at each time point
    """
    data = np.load(fname, allow_pickle=True)
    t = data['t']
    X = data['X']
    return t, X


# TODO: test this function
def normalize_lorenz96_2coupled(graph_tuple_dict):
    """ normalize dataset of GraphTuples using training data distribution.

        (replaced existing train, val, test with normalized versions)

    """
        # graph_tuple_dict has the following format:
        # {
        # 'train': {
        #     'inputs': list of graphtuples, which are batched window data
        #     'targets': list of graphtuples},
        # 'val': {
        #     'inputs': list of graphtuples,
        #     'targets': list of graphtuples},
        # 'test': {
        #     'inputs': list of graphtuples,
        #     'targets': list of graphtuples},
        # }

    # compute X1 mean and std, X2 mean and std, using solely input train data
    X1_input_nodes = []
    X2_input_nodes = []
    for window in graph_tuple_dict['train']['inputs']:
        for graphtuple in window: 
            X1_input_nodes.append(graphtuple.nodes[:, 0])
            X2_input_nodes.append(graphtuple.nodes[:, 1])    
    X1_input_nodes = np.concatenate(X1_input_nodes)
    X2_input_nodes = np.concatenate(X2_input_nodes)

    X1_mean = X1_input_nodes.mean()
    X2_mean = X2_input_nodes.mean()
    X1_std = X1_input_nodes.std()
    X2_std = X2_input_nodes.std()

    # normalize the data 
    # (we have to iterate over each graphtuple in the dataset anyway to extract the node features; kind of inefficient)
    for data_mode in ['train', 'val', 'test']:
        for data_type in ['inputs', 'targets']:
            for window in graph_tuple_dict[data_mode][data_type]:
                for i, graphtuple in enumerate(window): 
                    # normalize data 
                    norm_X1 = (graphtuple.nodes[:, 0] - X1_mean) / X1_std
                    norm_X2 = (graphtuple.nodes[:, 1] - X2_mean) / X2_std
                    # reassign data 
                    graphtuple = jraph.GraphsTuple(
                        globals=graphtuple.globals,
                        nodes=np.vstack((norm_X1, norm_X2)).T,
                        edges=graphtuple.edges,
                        receivers=graphtuple.receivers,
                        senders=graphtuple.senders,
                        n_node=graphtuple.n_node,
                        n_edge=graphtuple.n_edge)
                    
                    window[i] = graphtuple
                    
    return graph_tuple_dict