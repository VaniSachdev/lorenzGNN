################################################################################
# This file contains helper functions for generating and handling data from    #
# the Lorenz 96 model.                                                         #
################################################################################

# imports
import os

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

from datetime import datetime
import logging

DEFAULT_TIME_RESOLUTION = 100


# wrapper for the lorenzDataset with train/test splitting and normalization bundled in
class lorenzDatasetWrapper():

    def __init__(
            self,
            predict_from="X1X2_window",
            n_samples=10000,
            preprocessing=True,
            simple_adj=False,
            input_steps=2 * DEFAULT_TIME_RESOLUTION,  # 2 days
            output_delay=1 * DEFAULT_TIME_RESOLUTION,  # 1 day
            output_steps=1,
            min_buffer=10,
            rand_buffer=False,
            K=36,
            F=8,
            c=10,
            b=10,
            h=1,
            coupled=True,
            time_resolution=DEFAULT_TIME_RESOLUTION,
            init_buffer_steps=100,
            return_buffer=True,
            seed=42,
            override=False,
            train_pct=1.0,
            val_pct=0.0,
            test_pct=0.0):

        assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001
        # use error term due to float errors

        # set up variables
        self.predict_from = predict_from
        self.n_samples = int(n_samples)
        self.preprocessing = preprocessing
        self.K = K
        self.F = F
        self.c = c
        self.b = b
        self.h = h
        self.coupled = coupled
        self.time_resolution = int(time_resolution)
        self.init_buffer_steps = init_buffer_steps
        self.return_buffer = return_buffer
        self.seed = seed

        if self.predict_from == "X2":
            self.input_steps = 1
            self.output_steps = 1
            self.output_delay = 0
            self.min_buffer = 0
            self.rand_buffer = rand_buffer
        else:
            self.input_steps = int(input_steps)
            self.output_steps = int(output_steps)
            self.output_delay = int(output_delay)
            self.min_buffer = int(min_buffer)
            self.rand_buffer = rand_buffer

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct

        # generate dataset
        dataset_raw = lorenzDataset(predict_from=self.predict_from,
                                    n_samples=self.n_samples,
                                    preprocessing=self.preprocessing,
                                    simple_adj=simple_adj,
                                    input_steps=self.input_steps,
                                    output_delay=self.output_delay,
                                    output_steps=self.output_steps,
                                    min_buffer=self.min_buffer,
                                    rand_buffer=self.rand_buffer,
                                    K=self.K,
                                    F=self.F,
                                    c=self.c,
                                    b=self.b,
                                    h=self.h,
                                    coupled=self.coupled,
                                    time_resolution=self.time_resolution,
                                    init_buffer_steps=self.init_buffer_steps,
                                    return_buffer=self.return_buffer,
                                    seed=self.seed,
                                    override=override)
        # split dataset
        if init_buffer_steps > 0 and return_buffer:
            self.buffer = dataset_raw[:init_buffer_steps]
            dataset = dataset_raw[init_buffer_steps:]
        else:
            self.buffer = None
            dataset = dataset_raw

        train_bound = round(train_pct * dataset.n_graphs)
        val_bound = round((train_pct + val_pct) * dataset.n_graphs)

        self.train = dataset[:train_bound]
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

        self.X1_mean, self.X1_std, self.X2_mean, self.X2_std = norm_input.get_mean_std(
        )
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
            predict_from="X1X2_window",
            n_samples=10000,
            preprocessing=True,
            simple_adj=False,
            input_steps=2 * DEFAULT_TIME_RESOLUTION,  # 2 days
            output_delay=1 * DEFAULT_TIME_RESOLUTION,  # 1 day
            output_steps=1,
            min_buffer=10,
            rand_buffer=False,
            K=36,
            F=8,
            c=10,
            b=10,
            h=1,
            coupled=True,
            time_resolution=DEFAULT_TIME_RESOLUTION,
            init_buffer_steps=100,
            return_buffer=True,
            seed=42,
            override=False,
            **kwargs):
        """ Args: 
                predict_from (str): "X1X2_window", "X2_window", or "X2". 
                    indicates the structure of input/target data
                n_samples (int): sets of data samples to generate. (each sample 
                    contains <input_steps> steps of input data + <output_steps> 
                    steps of output data)
                preprocess (bool): whether or not to apply the gcn_filter 
                    preprocessing step to add self-loops and normalize the 
                    adjacency matrix. if False, the adjacency matrix will contain self-loops but won't be normalized. 
                input_steps (int): num of timesteps in each input window (only used if predict_from uses a window)
                output_steps (int): num of timesteps in each output window (only used if predict_from uses a window)
                output_delay (int): number of time_steps between end of input 
                    window and start of output window (only used if predict_from uses a window)
                min_buffer (int): min number of time_steps between end of output
                    window and start of input window (only used if predict_from uses a window)
                rand_buffer (bool): whether or not the buffer between sets of 
                    data will have a random or fixed length (only used if predict_from uses a window)
                K (int): number of points on the circumference
                F (float): forcing constant
                c (float): time-scale ratio ?
                b (float): spatial-scale ratio ?
                h (float): coupling parameter ?
                coupled (bool): whether to use the coupled 2-layer model or 
                    original 1-layer model
                time_resolution (float): number of timesteps per "day" in the simulation, i.e. inverse timestep for the ODE integration
                seed (int): for reproducibility 
                override (bool): whether or not to regenerate data that was 
                    already generated previously
        """
        self.preprocessing = preprocessing
        self.simple_adj = simple_adj
        self.predict_from = predict_from
        self.n_samples = int(n_samples)
        self.K = K
        self.F = F
        self.c = c
        self.b = b
        self.h = h
        self.coupled = coupled
        self.time_resolution = int(time_resolution)
        self.init_buffer_steps = init_buffer_steps
        self.return_buffer = return_buffer
        self.seed = seed
        self.a = None  # self.a is set to None anyway in super __init__ so we have to define it in the read() function

        if self.predict_from == "X2":
            self.input_steps = 1
            self.output_steps = 1
            self.output_delay = 0
            self.min_buffer = 0
            self.rand_buffer = rand_buffer
        else:
            self.input_steps = int(input_steps)
            self.output_steps = int(output_steps)
            self.output_delay = int(output_delay)
            self.min_buffer = int(min_buffer)
            self.rand_buffer = rand_buffer

        if override and os.path.exists(self.path):
            os.remove(self.path)

        # super().__init__(transforms=self.transforms,**kwargs)
        super().__init__(**kwargs)

    @property
    def path(self):
        """ define the file path where data will be stored/extracted. """
        drive_base_path = '/content/drive/My Drive/_research ML AQ/lorenz 96 gnn/lorenz_data'  # obviously this must change depending on your own computer's file system
        filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz".format(
            self.predict_from, self.n_samples, self.input_steps,
            self.output_steps, self.output_delay, self.min_buffer,
            self.rand_buffer, self.K, self.F, self.c, self.b, self.h,
            self.coupled, self.time_resolution, self.init_buffer_steps,
            self.return_buffer, self.seed)

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
        return dict(predict_from=self.predict_from,
                    n_samples=self.n_samples,
                    input_steps=self.input_steps,
                    output_steps=self.output_steps,
                    output_delay=self.output_delay,
                    min_buffer=self.min_buffer,
                    rand_buffer=self.rand_buffer,
                    K=self.K,
                    F=self.F,
                    c=self.c,
                    b=self.b,
                    h=self.h,
                    coupled=self.coupled,
                    time_resolution=self.time_resolution,
                    seed=self.seed)

    def read(self):
        """ reads stored dataset and returns a list of Graph objects. 

            assumes that the dataset file path already exists. (this is handled in super().__init__)
        """
        logging.info('reading Lorenz data from stored file')

        # create sparse adjacency matrix
        self.a = self.compute_adjacency_matrix()
        print('done computing adj')

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
            if self.init_buffer_steps > 0 and self.return_buffer:
                return [
                    Graph(x=X[i], y=Y[i], t=t[i])
                    for i in range(self.n_samples + self.init_buffer_steps)
                ]
            else:
                return [
                    Graph(x=X[i], y=Y[i], t=t[i]) for i in range(self.n_samples)
                ]

    def download(self):
        """ generate and store Lorenz data. """
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
        logging.info('generating window data')
        if self.rand_buffer:
            # TODO: maybe use an exponential distribution to determine buffer space
            raise NotImplementedError
        else:
            # equally spaced sets of samples
            x_windows = [
                (i * (self.min_buffer + self.input_steps + self.output_steps +
                      self.output_delay),
                 i * (self.min_buffer + self.input_steps + self.output_steps +
                      self.output_delay) + self.input_steps)
                for i in range(self.n_samples)
            ]
            y_windows = [
                (i * (self.min_buffer + self.input_steps + self.output_steps +
                      self.output_delay) + self.input_steps + self.output_delay,
                 i * (self.min_buffer + self.input_steps + self.output_steps +
                      self.output_delay) + self.input_steps +
                 self.output_steps + self.output_delay)
                for i in range(self.n_samples)
            ]

        # generate some data
        n_steps = y_windows[-1][1]
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
                init_buffer_steps=self.init_buffer_steps,
                return_buffer=self.return_buffer,
                seed=self.seed)
        else:
            raise NotImplementedError

        X = []
        Y = []
        t_X = []
        t_Y = []
        for i in range(self.n_samples):
            x_window = x_windows[i]
            y_window = y_windows[i]

            # originally, the dataframe has time indexing the rows and node features on the
            # columns; we want to reshape this so that we have rows indexed by node and
            # columns to contain the node feature at every time step

            # index and reshape dfs
            x_df = lorenz_buffered_df.iloc[x_window[0]:x_window[1]]
            t_x = x_df.index
            x_df = x_df.T
            x_df = pd.concat([
                x_df.iloc[:self.K].reset_index(drop=True),
                x_df.iloc[self.K:].reset_index(drop=True)
            ],
                             axis=1)

            # for y, we only want to predict the X1 values, not the X2 values
            y_df = lorenz_buffered_df.iloc[y_window[0]:y_window[1], :self.K]
            t_y = y_df.index
            y_df = y_df.T

            # rename columns
            x_df.columns = ['X1_{}'.format(t)
                            for t in t_x] + ['X2_{}'.format(t) for t in t_x]
            y_df.columns = ['X1_{}'.format(t) for t in t_y]

            # note that spektral graphs can't handle dataframes;
            # data must be in nparrays
            X.append(x_df.to_numpy())
            Y.append(y_df.to_numpy())
            t_X.append(t_x.to_numpy())
            t_Y.append(t_y.to_numpy())

        return X, Y, t_X, t_Y

    def generate_paired_data(self):
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
                init_buffer_steps=self.init_buffer_steps,
                return_buffer=self.return_buffer,
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
        """ Calculates the mean and stdev for 1) all X1 variables (includes both feature 
            and target data), and 2) for all X2 variables
        
            Returns:
                4-tuple: X1_mean, X1_std, X2_mean, X2_std
        """
        # get one mean/stdev for all X1 variables (includes the x and y data), and one mean/stdev for all X2 variables

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


def lorenzToDF(
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        coupled=True,
        n_steps=None,  # 30 * 100,
        n_days=30,
        time_resolution=100,
        init_buffer_steps=100,
        return_buffer=True,
        seed=42):
    """ generate a dataframe of data from the lorenz model. 

        Args: 
            K (int): number of points on the circumference
            F (float): forcing constant
            c (float): time-scale ratio ?
            b (float): spatial-scale ratio ?
            h (float): coupling parameter ?
            coupled (bool): whether to use the coupled 2-layer model or 
                original 1-layer model
            n_steps (int): number of timesteps to run the model for
            n_days (int): number of days to run the model for. Only used of 
                n_steps is None. 
            time_resolution (float): number of timesteps per "day" in the simulation, i.e. inverse timestep for the ODE integration. 
            seed (int): for reproducibility 
    """
    if n_steps is None:
        n_steps = n_days * time_resolution
    if coupled:
        t_buffered_raw, X_buffered_raw, _, _, _ = run_Lorenz96_2coupled(
            K=K,
            F=F,
            c=c,
            b=b,
            h=h,
            n_steps=n_steps,
            resolution=time_resolution,
            init_buffer_steps=init_buffer_steps,
            return_buffer=return_buffer,
            seed=seed)
    else:
        raise NotImplementedError

    df = pd.DataFrame(X_buffered_raw,
                      columns=['X1_{}'.format(i) for i in range(K)] +
                      ['X2_{}'.format(i) for i in range(K)],
                      index=t_buffered_raw)
    df.index.name = 'day'
    return df


#####################
# OG Lorenz96 model #
#####################
def lorenz96(X, t, K, F):
    """ (from Prof. Kavassalis)

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


def run_Lorenz96(K=36, F=8, number_of_days=30, nudge=True):
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
    """ (from Prof. Kavassalis)

        Args: 
            X (float array, size 2*K): array of current X and Y state values
            t: 
            K (int): number of points on the circumference
            F (float): forcing constant
            c (float): time-scale ratio ??
            b (float): spatial-scale ratio ??
            h (float): coupling parameter ??
        """
    dX_dt = np.zeros(K * 2)
    # dX/dt is the first K elements
    # dY/dt is the second K elements

    ######## first##########
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


def run_Lorenz96_2coupled(
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        n_steps=300,
        resolution=DEFAULT_TIME_RESOLUTION,  # 100
        n_days=None,
        init_buffer_steps=100,
        return_buffer=True,
        seed=42):
    """ (modified from Prof. Kavassalis) 
    
        note here that resolution = # of steps per day (e.g. 100 steps per 
        day), not the delta t (which would be 0.01)
    """
    random.seed(seed)

    # Initial state (equilibrium)
    X0 = np.concatenate((F * np.ones(K), (h * c / b) * np.ones(K)))

    # Perturbation
    X0[random.randint(0, K) -
       1] = X0[random.randint(0, K) - 1] + random.uniform(0, .01)

    if n_days is None:
        n_days = n_steps / resolution
    buffer_days = init_buffer_steps / resolution
    t_buffered = np.arange(0.0, buffer_days + n_days, 1 / resolution)

    logging.info('starting integration')
    X = odeint(lorenz96_2coupled,
               X0,
               t_buffered,
               args=(K, F, c, b, h),
               ixpr=True)

    if return_buffer:
        return t_buffered, X, F, K, n_days
    else:
        return t_buffered[init_buffer_steps:], X[
            init_buffer_steps:], F, K, n_days
