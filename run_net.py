# from utils.plotters import plot_data
from utils.jraph_data import lorenz_graph_tuple_list, print_graph_fts, get_data_windows, data_list_to_dict
# from utils.jraph_vis import draw_jraph_graph_structure, plot_time_series_for_node, plot_rollout_for_node, plot_predictions
from utils.jraph_models import MLPBlock_fn, MLPGraphNetwork_fn, naive_const_fn, naive_zero_fn
from utils.jraph_training import train, evaluate

# import jraph
# import jax
# import jax.numpy as jnp
# import networkx as nx
# import haiku as hk
# for training sequence
# import functools
# import optax
# from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

# import numpy as np
# import matplotlib.pyplot as plt

import os
import yaml
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
# TODO: add logging to file
# TODO: check AUTOFORMATTER

CFG_PATH = "configs/default_cfg.yaml"


def get_dataset(cfg):
    graph_tuple_lists = lorenz_graph_tuple_list(
        predict_from=cfg["PREDICTION_PARADIGM"], 
        n_samples=cfg["N_SAMPLES"], 
        input_steps=cfg["INPUT_STEPS"],
        output_delay=cfg["OUTPUT_DELAY"],
        output_steps=cfg["OUTPUT_STEPS"],
        min_buffer=cfg["MIN_BUFFER"],
        K=cfg["LORENZ"]["K"],
        F=cfg["LORENZ"]["F"],
        c=cfg["LORENZ"]["C"],
        b=cfg["LORENZ"]["B"],
        h=cfg["LORENZ"]["H"],
        coupled=cfg["LORENZ"]["COUPLED"],
        time_resolution=cfg["TIME_RESOLUTION"], 
        seed=cfg["SEED"],
        init_buffer_steps=cfg["BUFFER"]["INIT_BUFFER_STEPS"],
        return_buffer=cfg["BUFFER"]["RETURN_BUFFER"],
        train_pct=cfg["SPLIT"]["TRAIN_PCT"],
        val_pct=cfg["SPLIT"]["VAL_PCT"],
        test_pct=cfg["SPLIT"]["TEST_PCT"],
        override=cfg["OVERRIDE"]
    )

    n_rollout_steps = cfg["N_ROLLOUT_STEPS"]
    timestep_duration = cfg["TIMESTEP_DURATION"]
    data_dict_lists = {
        'train': data_list_to_dict(graph_tuple_lists['train'],
                        n_rollout_steps=n_rollout_steps,
                        timestep_duration=timestep_duration),
        'val': data_list_to_dict(graph_tuple_lists['val'],
                        n_rollout_steps=n_rollout_steps,
                        timestep_duration=timestep_duration),
        'test': data_list_to_dict(graph_tuple_lists['test'],
                        n_rollout_steps=n_rollout_steps,
                        timestep_duration=timestep_duration)
    }

    return data_dict_lists

def get_model_fn(cfg):
    model_arch = cfg["MODEL"]["ARCH"]

    model_fns = {"naive_zero": naive_zero_fn, 
                 "naive_const": naive_const_fn, 
                 "MLP_block": MLPBlock_fn, 
                 "MLP_graph_network": MLPGraphNetwork_fn}
    
    assert model_arch in model_fns.keys()
    return model_fns[model_arch]


def run(cfg):
    # load or generate dataset
    data_dict_lists = get_dataset(cfg)

    model_fn = get_model_fn(cfg)

    # run training pipeline
    assert cfg["TRAIN"]["ENABLE"] or (cfg["TRAIN"]["LOAD_PARAMS"] 
                                      and cfg["TRAIN"]["PARAM_PATH"] != "")
    if cfg["TRAIN"]["ENABLE"]:
        logging.info("running training pipeline")
        start = datetime.now()
        epochs = cfg["TRAIN"]["MAX_EPOCHS"]
        params = train(model_fn, 
                               data_dict_lists["train"], 
                               epochs=epochs)
        logging.info(f"training complete: runtime of {datetime.now() - start}")
    else: 
        # load existing parameters 
        raise NotImplementedError

    # run validation pipeline
    if cfg["VAL"]["ENABLE"]:
        logging.info("running validation pipeline")
        start = datetime.now()
        val_loss, val_preds = evaluate(model_fn, 
                                       data_dict_lists['val'],
                                       params)
        logging.info(f"validation complete: runtime of {datetime.now() - start}")

    # run testing pipeline
    if cfg["TEST"]["ENABLE"]:
        logging.info("running testing pipeline")
        start = datetime.now()
        test_loss, test_preds = evaluate(model_fn, 
                                       data_dict_lists['test'],
                                       params)
        print(test_loss)
        print(type(test_loss))
        print(test_preds)
        print(type(test_preds))
        logging.info(f"testing complete: runtime of {datetime.now() - start}")

    logging.info("run complete")
    


if __name__ == "__main__":
    with open(CFG_PATH, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict object

    run(cfg)
