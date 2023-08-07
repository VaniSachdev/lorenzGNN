import os
import json
import logging
import matplotlib.pyplot as plt
import pdb
import jax

def log_training_performance(cfg, epoch, mode, loss):
    # log the training and validation performance to a json file
    assert mode in ["train", "val"]

    log_path = get_log_path(cfg)

    # make sure the loss value is a float and not a jax array
    if isinstance(loss, jax.numpy.ndarray):
        loss = float(loss)

    performance_dict = {"epoch": epoch, "mode": mode, "avg_loss": loss}

    file_access_mode = "w" if (epoch == 0 and mode == "train") else "a"
    with open(log_path, file_access_mode) as f:
        json_str = json.dumps(performance_dict, indent=None)
        f.write(json_str + "\n")
        

def load_training_performance(cfg):
    """ load the training and validation performance from a json file
    
        Returns:
            a dictionary containing the training and validation losses
    """
    log_path = get_log_path(cfg)
    assert os.path.exists(log_path), "performance log file does not exist"

    losses_dict = {"train": [], "val": []}
    with open(log_path, "r") as f:
        for line in f:
            performance_dict = json.loads(line)
            losses_dict[performance_dict["mode"]].append(performance_dict["avg_loss"])
    
    return losses_dict

def get_performance_for_epoch(cfg, mode, epoch):
    # get the performance for a given mode and epoch
    losses_dict = load_training_performance(cfg)
    assert len(losses_dict[mode]) > epoch, f"epoch {epoch} does not exist in performance log"

    return losses_dict[mode][epoch]

def plot_training_performance(cfg):
    # plot the training and validation performance
    try:
        plot_path = os.path.join(cfg["OUTPUT_DIR"], "train_performance_curves.jpg")

        # retrieve data 
        losses_dict = load_training_performance(cfg)
        print(losses_dict)
        pdb.set_trace()

        # generate plot
        ax = plt.subplot(1, 1, 1)
        ax.plot(losses_dict["train"], "-o", label="train")
        ax.plot(losses_dict["val"], "-o", label="val")

        # add labels, titles, etc
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (MSE)") 
        ax.set_title("Loss over training")
        ax.legend()
        plt.tight_layout()

        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        logging.warning(f'Failed to plot train/val curves with Exception {e}')

def get_log_path(cfg):
    # get the path to the log file and create the directory if it doesn't exist
    log_path = os.path.join(cfg["OUTPUT_DIR"], "training_performance.log")
    if not os.path.exists(cfg["OUTPUT_DIR"]):
        os.makedirs(cfg["OUTPUT_DIR"])
    return log_path