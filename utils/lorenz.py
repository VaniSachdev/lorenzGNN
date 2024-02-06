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
import jraph 
import jax.numpy as jnp

from datetime import datetime
import logging
import pdb

DEFAULT_TIME_RESOLUTION = 100
DATA_DIRECTORY_PATH = "/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/data/data_directory.json"


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
            c (float): Lorenz96 time-scale ratio
            b (float): Lorenz96 spatial-scale ratio
            h (float): Lorenz96 coupling parameter
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
                X = data['X'] # array of state values at each time point, shape (n_steps, K*2)
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