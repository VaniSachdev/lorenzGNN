from utils.plotters import plot_data
from utils.jraph_data import lorenz_graph_tuple_list, print_graph_fts, get_data_windows, data_list_to_dict
from utils.jraph_vis import draw_jraph_graph_structure, plot_time_series_for_node, plot_rollout_for_node, plot_predictions
from utils.jraph_models import MLPBlock_fn, MLPGraphNetwork_fn, naive_const_fn, naive_zero_fn
from utils.jraph_training import train, evaluate

import jraph
import jax
import jax.numpy as jnp
import networkx as nx
import haiku as hk
# for training sequence
import functools
import optax
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
# TODO: add logging to file
# TODO: check AUTOFORMATTER

CFG_PATH = "/research/cwloka/projects/rohit_sandbox/sequence_generator/configs/sequence_generator.yaml"


def get_dataset(cfg):
    graph_tuple_lists = lorenz_graph_tuple_list(n_samples=200)


def train(dataset, cfg):

    def train_epoch():
        pass

    pass


def val(dataset, cfg):
    pass


def test(dataset, cfg):
    pass


def run(cfg):
    # load or generate dataset
    dataset = get_dataset(cfg)

    # run training pipeline
    if cfg["TRAIN"]["ENABLE"]:
        train(dataset, cfg)

    # run validation pipeline
    if cfg["VAL"]["ENABLE"]:
        val(dataset, cfg)

    # run testing pipeline
    if cfg["TEST"]["ENABLE"]:
        test(dataset, cfg)

    # TODO: log everything complete


if __name__ == "__main__":
    with open(CFG_PATH, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict object

    run(cfg)
