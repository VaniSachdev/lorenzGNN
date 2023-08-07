from utils.checkpoints import Checkpoint, load_checkpoint_from_path, load_latest_checkpoint, get_latest_checkpoint_path
from run_net import main as run_net_main, get_dataset, set_up_logging
from utils.jraph_training import evaluate
from utils.logging import get_performance_for_epoch
import yaml
import os
import unittest
import logging
from datetime import datetime
import pdb


class CheckpointTests(unittest.TestCase):
    """ test suite for checkpoint implementation. """

    def setUp(self) -> None:
        # load config and set up logging
        self.cfg_path = "configs/code_testing.yaml"
        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict object

        set_up_logging(self.cfg)

    def test_checkpoints(self):
        logging.info('\n test_checkpoints \n')

        # load dataset for later testing
        dataset_dict_list = get_dataset(self.cfg)

        # run a small training loop
        run_net_main(cfg_path=self.cfg_path)

        # check that checkpoints are saved for all epochs
        checkpoint_dir = os.path.join(self.cfg["OUTPUT_DIR"], "checkpoints")
        epochs = self.cfg["TRAIN"]["MAX_EPOCHS"]

        for epoch in range(epochs):
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f"checkpoint_epoch_{epoch:06d}.pkl")
            self.assertTrue(os.path.exists(checkpoint_path))

        # test load_checkpoint from specific path
        checkpoint_2_path = os.path.join(checkpoint_dir,
                                         f"checkpoint_epoch_{2:06d}.pkl")
        checkpoint_2 = load_checkpoint_from_path(checkpoint_2_path)

        # run validation and verify the performance is the same as logged
        true_epoch_2_val_loss = get_performance_for_epoch(self.cfg, "val", 2)
        loss_2, _ = evaluate(net_fn=checkpoint_2.net_fn,
                             val_dataset=dataset_dict_list["val"],
                             params=checkpoint_2.model_params)
        self.assertAlmostEqual(loss_2, true_epoch_2_val_loss)

        # test load_checkpoint from latest path
        checkpoint_last = load_latest_checkpoint(self.cfg)

        # run validation and verify the performance is the same as logged
        true_epoch_last_val_loss = get_performance_for_epoch(
            self.cfg, "val", self.cfg["TRAIN"]["MAX_EPOCHS"] - 1)
        loss_last, _ = evaluate(net_fn=checkpoint_last.net_fn,
                                val_dataset=dataset_dict_list["val"],
                                params=checkpoint_last.model_params)
        self.assertAlmostEqual(loss_last, true_epoch_last_val_loss)

        # TODO: test load checkpoint by mode in config
        pass


class StatsLogTest(unittest.TestCase):

    def test_stats_log(self):
        pass


if __name__ == "__main__":
    # set up logging for unittest outputs
    log_file = f"tests/outputs/checkpoint_tests_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    with open(log_file, "w") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner)
