import os

def get_checkpoint(mode, cfg):
    """ Get a checkpoint according to the config file. 
    
        Args: 
            mode (str): "train", "val", or "test"
            cfg (dict): yaml config dictionary

        Returns: 
            a checkpoint (type?) if a valid one exists, otherwise None
    """
    assert mode in ["train", "val", "test"]

    checkpoint_dir = os.path.join(cfg["OUTPUT_DIR"], "checkpoints")

    if mode == "train" and not cfg["TRAIN"]["RESUME_FROM_CHECKPOINT"]:
        return None
        
    if "CHECKPOINT" in cfg[mode].keys() and cfg[mode]["CHECKPOINT"] != "":
        # load checkpoint from explicit path
        checkpoint_path = cfg[mode]["CHECKPOINT"]
        return load_checkpoint_from_path(checkpoint_path)

    else: # load latest epoch
        return load_latest_checkpoint(cfg)
    

def load_checkpoint_from_path(path):
    """ Load a checkpoint from an explicit path. 

        Args:
            path (str): path to checkpoint file

        Returns: 
            a checkpoint (type?) 
    """
    # TODO
    return None

def load_latest_checkpoint(cfg):
    """ Load a checkpoint implicitly from the latest epoch. 

        Returns: 
            a checkpoint (type?) if a valid one exists, otherwise None
    """
    checkpoint_path = get_latest_checkpoint_path(cfg)
    return load_checkpoint_from_path(checkpoint_path)

def save_checkpoint(cfg):
    """ Save a checkpoint. File name is defined automatically by epoch. 

        Args: 
            cfg (dict): yaml config dictionary
    """
    # checkpoint_dir = os.path.join(cfg["OUTPUT_DIR"], "checkpoints")
    next_checkpoint_path = get_next_checkpoint_path(cfg)
    
    pass

def get_latest_checkpoint_path(cfg):
    """ Retrieves the path to the checkpoint of the latest epoch. 
    
        Args: 
            cfg (dict): yaml config dictionary

        Returns: 
            checkpoint path string if a valid one exists, otherwise None
    """
    checkpoint_dir = os.path.join(cfg["OUTPUT_DIR"], "checkpoints")
    pass

def get_next_checkpoint_path(cfg):
    """ Creates the path to the next checkpoint. 
    
        Args: 
            cfg (dict): yaml config dictionary

        Returns: 
            checkpoint path string
    """
    pass
