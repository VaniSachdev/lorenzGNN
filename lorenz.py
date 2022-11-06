# imports
import numpy as np
import random
from scipy.integrate import odeint
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.data.dataset import Dataset


# create dataset class for lorenz96 model
class lorenzDataset(Dataset):
    """ A dataset containing windows of data from a Lorenz96 time series. """

    def __init__(
            self,
            n_samples=42,
            input_steps=50,  # eventually we want something on the order of 1000s; this is small for now for quick prototyping
            output_steps=1,
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
        # TODO: update docstrings
        """ Args: 
                n_samples (int): sets of data samples to generate. (each sample 
                    contains <input_steps> steps of input data + <output_steps> 
                    steps of output data)
                input_steps (int): num of timesteps in each input data sample
                output_steps (int): num of timesteps in each output data sample
                min_buffer (int): min number of time_steps between each set of 
                    data samples
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
        self.n_samples = n_samples
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.min_buffer = min_buffer
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

        # generate a sequence guide thing to determine how our samples will be
        # spaced out
        # x_guide is a list of <n_samples> tuples; each element is a tuple
        # containing the first (inclusive) and last (non-inclusive) indices of
        # the input data sample
        # y_guide is the same but for the output data sample
        if self.rand_buffer:
            # TODO: maybe use an exponential distribution to determine buffer space
            raise NotImplementedError
        else:
            # equally spaced sets of samples
            x_guide = [
                (i * (self.min_buffer + self.input_steps + self.output_steps),
                 i * (self.min_buffer + self.input_steps + self.output_steps) +
                 self.input_steps) for i in range(self.n_samples)
            ]
            y_guide = [
                (i * (self.min_buffer + self.input_steps + self.output_steps) +
                 self.input_steps,
                 i * (self.min_buffer + self.input_steps + self.output_steps) +
                 self.input_steps + self.output_steps)
                for i in range(self.n_samples)
            ]

        # generate some data
        n_steps = y_guide[-1][1]
        if self.coupled:
            t_raw, X_raw, _, _, _ = run_Lorenz96_2coupled(
                K=self.K,
                F=self.F,
                c=self.c,
                b=self.b,
                h=self.h,
                n_steps=n_steps,
                resolution=self.time_resolution,
                seed=self.seed)
        else:
            raise NotImplementedError
        # X_raw has shape [n_steps, K * 2]
        # the first K columns in X_raw are the X1 (e.g. atmospheric) variable
        # from the Lorenz model; the last K columns in X_raw are the X2 (e.g.
        # oceanic) variable from the Lorenz model

        X = np.stack([X_raw[sample[0]:sample[1]] for sample in x_guide])
        Y = np.stack([X_raw[sample[0]:sample[1]] for sample in y_guide])
        t_X = np.array([t_raw[sample[0]:sample[1]] for sample in x_guide])
        t_Y = np.array([t_raw[sample[0]:sample[1]] for sample in y_guide])
        # X has shape (n_samples, input_steps, K * 2)
        # Y has shape (n_samples, output_steps, K * 2)

        # reshape X to have shape (n_samples, n_nodes, n_node_features)
        # = (n_samples, K, 2 * input_steps)
        # and reshape Y to have shape (n_samples, K, output_steps)
        X1, X2 = np.split(X, 2, axis=2)
        X_reshaped = np.concatenate((X1, X2), axis=1)
        X_reshaped = np.swapaxes(X_reshaped, 1, 2)
        Y_reshaped = np.swapaxes(Y[:, :, :self.K], 1, 2)

        # convert to Graph structure
        return [
            Graph(x=X_reshaped[i], y=Y_reshaped[i], t_X=t_X[i], t_Y=t_Y[i])
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
