from utils.jraph_data import lorenz_graph_tuple_list, data_list_to_dict
from utils.jraph_models import MLPBlock_fn, MLPGraphNetwork_fn, naive_const_fn, naive_zero_fn
from utils.jraph_training import train, evaluate

import os
import yaml
from datetime import datetime
import logging
import pdb

CFG_PATH = "configs/default_cfg.yaml"


def get_dataset(cfg):
    graph_tuple_lists = lorenz_graph_tuple_list(
        predict_from=cfg["DATA"]["PREDICTION_PARADIGM"],
        n_samples=cfg["DATA"]["N_SAMPLES"],
        input_steps=cfg["DATA"]["INPUT_STEPS"],
        output_delay=cfg["DATA"]["OUTPUT_DELAY"],
        output_steps=cfg["DATA"]["OUTPUT_STEPS"],
        min_buffer=cfg["DATA"]["MIN_BUFFER"],
        K=cfg["DATA"]["LORENZ"]["K"],
        F=cfg["DATA"]["LORENZ"]["F"],
        c=cfg["DATA"]["LORENZ"]["C"],
        b=cfg["DATA"]["LORENZ"]["B"],
        h=cfg["DATA"]["LORENZ"]["H"],
        coupled=cfg["DATA"]["LORENZ"]["COUPLED"],
        time_resolution=cfg["DATA"]["TIME_RESOLUTION"],
        seed=cfg["DATA"]["SEED"],
        init_buffer_steps=cfg["DATA"]["BUFFER"]["INIT_BUFFER_STEPS"],
        return_buffer=cfg["DATA"]["BUFFER"]["RETURN_BUFFER"],
        train_pct=cfg["DATA"]["SPLIT"]["TRAIN_PCT"],
        val_pct=cfg["DATA"]["SPLIT"]["VAL_PCT"],
        test_pct=cfg["DATA"]["SPLIT"]["TEST_PCT"],
        override=cfg["DATA"]["OVERRIDE"])

    n_rollout_steps = cfg["DATA"]["N_ROLLOUT_STEPS"]
    timestep_duration = cfg["DATA"]["TIMESTEP_DURATION"]
    data_dict_lists = {
        'train':
        data_list_to_dict(graph_tuple_lists['train'],
                          n_rollout_steps=n_rollout_steps,
                          timestep_duration=timestep_duration),
        'val':
        data_list_to_dict(graph_tuple_lists['val'],
                          n_rollout_steps=n_rollout_steps,
                          timestep_duration=timestep_duration),
        'test':
        data_list_to_dict(graph_tuple_lists['test'],
                          n_rollout_steps=n_rollout_steps,
                          timestep_duration=timestep_duration)
    }

    return data_dict_lists


def get_model_fn(cfg):
    model_arch = cfg["MODEL"]["ARCH"]

    model_fns = {
        "naive_zero": naive_zero_fn,
        "naive_const": naive_const_fn,
        "MLP_block": MLPBlock_fn,
        "MLP_graph_network": MLPGraphNetwork_fn
    }

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
                       data_dict_lists=data_dict_lists,
                       epochs=epochs,
                       cfg=cfg)
        logging.info(f"training complete: runtime of {datetime.now() - start}")
    else:
        # load existing parameters
        raise NotImplementedError

    # run validation pipeline
    if cfg["VAL"]["ENABLE"]:
        logging.info("running validation pipeline")
        start = datetime.now()
        val_loss, val_preds = evaluate(model_fn, data_dict_lists['val'], params)
        logging.info(
            f"validation complete: runtime of {datetime.now() - start}")

    # run testing pipeline
    if cfg["TEST"]["ENABLE"]:
        logging.info("running testing pipeline")
        start = datetime.now()
        test_loss, test_preds = evaluate(model_fn, data_dict_lists['test'],
                                         params)
        logging.info(f"testing complete: runtime of {datetime.now() - start}")

    logging.info("run complete")


def set_up_logging(cfg):
    log_dir = os.path.join(cfg["OUTPUT_DIR"], "logs")
    log_path = os.path.join(
        log_dir, f"stdout_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }[cfg["LOG_LEVEL"]]
    # TODO: check it it creates the directory if it doesn't exist

    logging.basicConfig(filename=log_path, encoding='utf-8', level=log_level)


def main(cfg_path):
    # load config
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict object

    # set up logging
    set_up_logging(cfg)

    # print config
    logging.info(f"config:\n{yaml.dump(cfg)} \n--------------------\n")

    # run
    run(cfg)


if __name__ == "__main__":
    # cfg_path = CFG_PATH  # change as desired
    cfg_path = "configs/code_testing.yaml"

    main(cfg_path=cfg_path)