# imports
import numpy as np
import pandas as pd
import random
from scipy.integrate import odeint
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.data.dataset import Dataset


# create dataset class for lorenz96 model
class lorenzDataset(Dataset):
    """ A dataset containing windows of data from a Lorenz96 time series. """

    def __init__(self,
                 n_samples=42,
                 input_steps=50,
                 output_steps=1,
                 output_delay=4,
                 min_buffer=4,
                 rand_buffer=False,
                 K=36,
                 F=8,
                 c=10,
                 b=10,
                 h=1,
                 coupled=True,
                 time_resolution=0.01,
                 seed=42,
                 **kwargs):
        """ Args: 
                n_samples (int): sets of data samples to generate. (each sample 
                    contains <input_steps> steps of input data + <output_steps> 
                    steps of output data)
                input_steps (int): num of timesteps in each input window
                output_steps (int): num of timesteps in each output window
                output_delay (int): number of time_steps between end of input 
                    window and start of output window
                min_buffer (int): min number of time_steps between end of output
                    window and start of input window
                rand_buffer (bool): whether or not the buffer between sets of 
                    data will have a random or fixed length
                K (int): number of points on the circumference
                F (float): forcing constant
                c (float): time-scale ratio ?
                b (float): spatial-scale ratio ?
                h (float): coupling parameter ?
                coupled (bool): whether to use the coupled 2-layer model or 
                    original 1-layer model
                time_resolution (float): timestep for the ODE integration, i.e. 
                    inverse number of timesteps per "day" in the simulation
                seed (int): for reproducibility 
        """
        self.a = None  # adjacency list
        self.n_samples = int(n_samples)
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.output_delay = int(output_delay)
        self.min_buffer = int(min_buffer)
        self.rand_buffer = rand_buffer
        self.K = K
        self.F = F
        self.c = c
        self.b = b
        self.h = h
        self.coupled = coupled
        self.time_resolution = time_resolution
        self.seed = seed
        super().__init__(**kwargs)

    def read(self):
        """ returns a list of Graph objects. """
        # create adjacency list
        self.a = self.compute_adjacency_matrix()

        # generate a sequence of windows to determine how our samples will be
        # spaced out
        # x_windows is a list of <n_samples> tuples; each element is a tuple
        # containing the first (inclusive) and last (non-inclusive) indices of
        # the input data sample
        # y_windows is the same but for the output data sample
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
                      self.output_delay) + self.input_steps +
                 self.output_delay,
                 i * (self.min_buffer + self.input_steps + self.output_steps +
                      self.output_delay) + self.input_steps +
                 self.output_steps + self.output_delay)
                for i in range(self.n_samples)
            ]

        # generate some data
        n_steps = y_windows[-1][1]
        if self.coupled:
            lorenz_df = lorenzToDF(K=self.K,
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
        for i in range(self.n_samples):
            x_window = x_windows[i]
            y_window = y_windows[i]

            # originally, the dataframe has time indexing the rows and node features on the
            # columns; we want to reshape this so that we have rows indexed by node and
            # columns to contain the node feature at every time step

            # index and reshape dfs
            x_df = lorenz_df.iloc[x_window[0]:x_window[1]]
            t_x = x_df.index
            x_df = x_df.T
            x_df = pd.concat([
                x_df.iloc[:self.K].reset_index(drop=True),
                x_df.iloc[self.K:].reset_index(drop=True)
            ],
                             axis=1)

            # for y, we only want to predict the X1 values, not the X2 values
            y_df = lorenz_df.iloc[y_window[0]:y_window[1], :self.K]
            t_y = y_df.index
            y_df = y_df.T

            # rename columns
            x_df.columns = ['X1_{}'.format(t)
                            for t in t_x] + ['X2_{}'.format(t) for t in t_x]
            y_df.columns = ['X1_{}'.format(t) for t in t_y]

            # note that spektral graphs can't handle dataframes; data must be in nparrays
            X.append(x_df.to_numpy())
            Y.append(y_df.to_numpy())
            t_X.append(t_x.to_numpy())
            t_Y.append(t_y.to_numpy())

        # TODO: add sinusoids for time of day as an engineered feature

        # convert to Graph structure
        return [
            Graph(x=X[i], y=Y[i], t_X=t_X[i], t_Y=t_Y[i])
            for i in range(self.n_samples)
        ]

    def compute_adjacency_matrix(self):
        src_nodes = np.concatenate(
            (np.arange(self.K), np.arange(self.K), np.arange(self.K)))
        target_nodes = np.mod(
            np.concatenate((np.arange(self.K), (np.arange(self.K) - 1),
                            (np.arange(self.K) + 1))), self.K)
        weights = np.ones(shape=len(src_nodes))
        return coo_matrix((weights, (src_nodes, target_nodes)),
                          shape=(self.K, self.K))

    def get_mean(self):
        """ Calculates the mean and stdev for 1) all X1 variables (includes both feature 
            and target data), and 2) for all X2 variables
        
            Returns:
                4-tuple: X1_mean, X1_std, X2_mean, X2_std
        """
        # get one mean/stdev for all X1 variables (includes the x and y data), and one mean/stdev for all X2 variables

        all_x =  np.concatenate([g.x for g in self])
        all_y =  np.concatenate([g.y for g in self])

        # print(all_x[:, :all_x.shape[1]//2].shape)
        # print(all_y.shape)
        # print(all_x[:, :all_x.shape[1]//2])
        # print(all_y)
        # print(np.concatenate([all_x[:, :all_x.shape[1]//2], all_y], axis=1))

        X1_mean = np.concatenate([all_x[:, :all_x.shape[1]//2], all_y], axis=1).mean()
        X1_std = np.concatenate([all_x[:, :all_x.shape[1]//2], all_y], axis=1).std()
        X2_mean = all_x[:, all_x.shape[1]//2 :].mean()
        X2_std = all_x[:, all_x.shape[1]//2 :].std()

        return X1_mean, X1_std, X2_mean, X2_std

    def normalize(self, mean, std):
        pass

def lorenzToDF(
        K=36,
        F=8,
        c=10,
        b=10,
        h=1,
        coupled=True,
        n_steps=None,  # 30/0.01,
        n_days=30,
        time_resolution=0.01,
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
            time_resolution (float): timestep for the ODE integration, i.e. 
                inverse number of timesteps per "day" in the simulation. Only 
                used of n_steps is None. 
            seed (int): for reproducibility 
    """
    if n_steps is None:
        n_steps = n_days / time_resolution
    if coupled:
        t_raw, X_raw, _, _, _ = run_Lorenz96_2coupled(
            K=K,
            F=F,
            c=c,
            b=b,
            h=h,
            n_steps=n_steps,
            #   resolution=time_resolution,
            seed=seed)
    else:
        raise NotImplementedError

    df = pd.DataFrame(X_raw,
                      columns=['X1_{}'.format(i) for i in range(K)] +
                      ['X2_{}'.format(i) for i in range(K)],
                      index=t_raw)
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

    X = odeint(lorenz96, X0, t,
               args=(K,
                     F))  #solves the system of ordinary differential equations

    return t, X, F, K, number_of_days  #gives us the output


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


def run_Lorenz96_2coupled(K=36,
                          F=8,
                          c=10,
                          b=10,
                          h=1,
                          n_steps=300,
                          resolution=0.01,
                          number_of_days=None,
                          seed=42):
    """ (from Prof. Kavassalis) """
    random.seed(seed)

    # Initial state (equilibrium)
    X0 = np.concatenate((F * np.ones(K), (h * c / b) * np.ones(K)))

    # Perturbation
    X0[random.randint(0, K) -
       1] = X0[random.randint(0, K) - 1] + random.uniform(0, .01)

    if number_of_days is None:
        number_of_days = n_steps * resolution
    t = np.arange(0.0, number_of_days, 0.01)
    X = odeint(lorenz96_2coupled, X0, t, args=(K, F, c, b, h))

    return t, X, F, K, number_of_days
