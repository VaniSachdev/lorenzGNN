import os
import pickle
import logging
import pdb

class Checkpoint(object):

    def __init__(self, epoch, net_fn, model_params, opt_state, cfg):
        """ Initialize a Checkpoint object.  

            Args: 
                epoch (int): epoch number
                net_fn (function): jax model function
                model_params (dict): jax model params
                opt_state (dict): jax optimizer state
                cfg (dict): yaml config dictionary
        """
        self.epoch = epoch
        self.net_fn = net_fn
        self.model_params = model_params
        self.opt_state = opt_state
        self.cfg = cfg

    def save_checkpoint(self):
        """ Save a checkpoint. File name is defined automatically by epoch. 
        """
        make_checkpoint_dir(self.cfg)
        next_checkpoint_path = get_path_to_checkpoint( self.cfg, self.epoch)
        with open(next_checkpoint_path, "wb") as f:
            pickle.dump(self, f)

    # # static overload function for load_checkpoint()
    # @staticmethod
    # def load_checkpoint(cfg, mode):
    #     return load_checkpoint(cfg, mode)


    
def load_checkpoint(cfg, mode):
    """ Get a checkpoint according to the config file. If no checkpoint is 
        specified, loads the checkpoint from the latest epoch.        
        Args: 
            cfg (dict): yaml config dictionary
            mode (str): "train", "val", or "test"
        Returns: 
            a Checkpoint object
    """
    assert mode in ["train", "val", "test"]

    if mode == "train" and not cfg["TRAIN"]["RESUME_FROM_CHECKPOINT"]:
        return None

    if "CHECKPOINT" in cfg[mode].keys(
    ) and cfg[mode]["CHECKPOINT"] != "":
        # load checkpoint from explicit path
        checkpoint_path = cfg[mode]["CHECKPOINT"]
        return load_checkpoint_from_path(checkpoint_path)

    else:  # load latest epoch
        return load_latest_checkpoint(cfg)

def load_checkpoint_from_path(path):
    """ Load a checkpoint from an explicit path. 
        Args:
            path (str): path to checkpoint file
        Returns: 
            a Checkpoint object
    """
    assert os.path.exists(path), "checkpoint path does not exist"
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint

def load_latest_checkpoint(cfg):
    """ Load a checkpoint implicitly from the latest epoch. 
        Args:
            cfg (dict): yaml config dictionary
        Returns: 
            a Checkpoint object
    """
    checkpoint_path = get_latest_checkpoint_path(cfg)
    if checkpoint_path is None:
        raise Exception(
            "no checkpoints exist, cannot load latest checkpoint")
    return load_checkpoint_from_path(checkpoint_path)

def get_latest_checkpoint_path(cfg):
    """ Retrieves the path to the checkpoint of the latest epoch. 
        Args:
            cfg (dict): yaml config dictionary
        Returns: 
            checkpoint path string if a valid one exists, otherwise None
    """
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_fnames = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")
    ]

    if len(checkpoint_fnames) == 0:
        return None

    # Sort the checkpoints by epoch.
    last_checkpoint_fname = sorted(checkpoint_fnames)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_fname)


def get_path_to_checkpoint(cfg, epoch):
    """ Get the full path to a checkpoint file based on epoch being saved.
        Args:
            cfg (dict): yaml config dictionary
            epoch (int): epoch number
        Returns: 
            checkpoint path string
   """
    checkpoint_dir = get_checkpoint_dir(cfg)
    fname = f"checkpoint_epoch_{epoch:06d}.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, fname)

    assert os.path.exists(checkpoint_path), f"checkpoint path does not exist for epoch {epoch}"
    return checkpoint_path

def make_checkpoint_dir(cfg):
    """ Creates the checkpoint directory if it does not already exist. 
        Args:
            cfg (dict): yaml config dictionary 
    """
    checkpoint_dir = get_checkpoint_dir(cfg)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def get_checkpoint_dir(cfg):
    """ Get path for storing checkpoints.
        Args:
            cfg (dict): yaml config dictionary 
        Returns:
            checkpoint directory path
    """
    checkpoint_dir = os.path.join(cfg["OUTPUT_DIR"], "checkpoints")
    return checkpoint_dir
