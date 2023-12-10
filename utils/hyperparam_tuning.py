from utils.jraph_training import train_and_evaluate_with_data, create_dataset
# from utils.jraph_models import MLPGraphNetwork
import ml_collections
import optuna 
from functools import partial
from datetime import datetime
import os 

CHECKPOINT_PATH = "/Users/h.lu/Documents/_code/_research lorenz code/lorenzGNN/experiments/tuning"

# TODO finish this 
def get_base_config():
    config = ml_collections.ConfigDict()

    return config

def objective(trial, datasets):
    """ Defines the objective function to be optimized over, aka the validation loss of a model.
    
        Args:
            trial: object which characterizes the current run 
            datasets: dictionary of data. we explicitly pass this in so that we don't have to waste runtime regenerating the same dataset over and over. 
    """
    # create config 
    config = get_base_config()

    # TODO replace these values during tuning 

    # Optimizer.
    config.optimizer = "adam"
    # config.optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, 
                                               log=True)
    if config.optimizer == "sgd":
        config.momentum = trial.suggest_float('momentum', 0, 0.999) # upper bound is inclusive, and we want to exclude a momentum of 1 because that would yield no decay 

    # Data params that are used in training 
    config.output_steps = 4

    # Training hyperparameters.
    config.batch_size = 1 # variable currently not used
    config.epochs = 5
    config.log_every_epochs = 5
    config.eval_every_epochs = 5
    config.checkpoint_every_epochs = 10

    # GNN hyperparameters.
    config.model = 'MLPGraphNetwork'
    config.n_blocks = trial.suggest_int("n_blocks", 1, 10)
    config.share_params = False
    config.dropout_rate = trial.suggest_float('dropout_rate', 0, 0.6)
    config.skip_connections = False # This was throwing a broadcast error in add_graphs_tuples_nodes when this was set to True
    config.layer_norm = False # TODO perhaps we want to turn on later

    # choose the hidden layer feature size using powers of 2 
    config.edge_features = (
        2**trial.suggest_int("edge_mlp_1_power", 1, 5), # range 2 - 64; upper bound is inclusive
        2**trial.suggest_int("edge_mlp_2_power", 1, 5), # range 2 - 64
    )
    config.node_features = (
        2**trial.suggest_int("node_mlp_1_power", 1, 9), # range 2 - 512
        # 2**trial.suggest_int("node_mlp_2_power", 1, 9), # range 2 - 512
        2) 
    # note the last feature size will be the number of features that the graph predicts
    config.global_features = None

    # generate a workdir 
    # TODO: check if we actually care about referencing this in the future or if we can just create a temp dir 
    workdir=os.path.join(CHECKPOINT_PATH, str(datetime.now()))

    # run training 
    state, train_metrics, eval_metrics_dict = train_and_evaluate_with_data(config=config, workdir=workdir, datasets=datasets)
    
    # retrieve and return val loss (MSE)
    print("eval_metrics_dict['val'].loss", eval_metrics_dict['val'].loss)
    print()
    return eval_metrics_dict['val'].loss.total / eval_metrics_dict['val'].loss.count


def get_data_config():
    config = get_base_config()

    config.n_samples=100
    config.input_steps=1
    config.output_delay=8 # predict 24 hrs into the future 
    config.output_steps=4
    config.timestep_duration=3 # equivalent to 3 hours
    # note a 3 hour timestep resolution would be 5*24/3=40
    # if the time_resolution is 120, then a sampling frequency of 3 would achieve a 3 hour timestep 
    config.sample_buffer = -1 * (config.input_steps + config.output_delay + config.output_steps - 1) # negative buffer so that our sample input are continuous (i.e. the first sample would overlap a bit with consecutive samples) 
        # number of timesteps strictly between the end 
        # of one full sample and the start of the next sample
    config.time_resolution=120 # the number of 
                # raw data points generated per time unit, equivalent to the 
                # number of data points generated per 5 days in the simulation
    config.init_buffer_samples=100
    config.train_pct=0.7
    config.val_pct=0.2
    config.test_pct=0.1
    config.K=36
    config.F=8
    config.c=10
    config.b=10
    config.h=1
    config.seed=42
    config.normalize=True

    return config

def prepare_study(study_name):
    # run optimization study
    db_path = os.path.join(CHECKPOINT_PATH, study_name, "optuna_hparam_search.db")
    if not os.path.exists(os.path.join(CHECKPOINT_PATH, study_name)):
        os.makedirs(os.path.join(CHECKPOINT_PATH, study_name))

    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{db_path}', # generates a new db if it doesn't exist
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=1,
            ), 
        load_if_exists=True, 
    )
    
    return study

def get_objective_with_dataset():
    # generate dataset 
    dataset_config = get_data_config()
    datasets = create_dataset(dataset_config)

    # get the objective function that reuses the pre-generated datasets 
    objective_with_dataset = partial(objective, datasets=datasets)

    return objective_with_dataset


def get_best_trial_config(study):
    dataset_config = get_data_config()
    best_trial_config = dataset_config

    # Optimizer.
    best_trial_config.optimizer = study.best_params['optimizer']
    best_trial_config.learning_rate = study.best_params['learning_rate']
    if best_trial_config.optimizer == "sgd":
        best_trial_config.momentum = study.best_params['momentu,']

    # Training hyperparameters.
    # best_trial_config.batch_size = 1 # variable currently not used
    # best_trial_config.epochs = 10
    # best_trial_config.log_every_epochs = 5
    # best_trial_config.eval_every_epochs = 5
    # best_trial_config.checkpoint_every_epochs = 10

    # GNN hyperparameters.
    best_trial_config.model = 'MLPBlock'
    best_trial_config.dropout_rate = study.best_params['dropout_rate']
    best_trial_config.skip_connections = False # This was throwing a broadcast error in add_graphs_tuples_nodes when this was set to True
    best_trial_config.layer_norm = False # TODO perhaps we want to turn on later
    best_trial_config.activation = study.best_params['activation']

    # choose the hidden layer feature size using powers of 2 
    best_trial_config.edge_features = (
        2**study.best_params["edge_mlp_1_power"],
        2**study.best_params["edge_mlp_2_power"],
    )
    best_trial_config.node_features = (
        2**study.best_params["node_mlp_1_power"],
        2**study.best_params["node_mlp_2_power"],
        2) 
    # note the last feature size will be the number of features that the graph predicts
    best_trial_config.global_features = None

    return best_trial_config

def get_best_trial_workdir(study):
    workdir=os.path.join(CHECKPOINT_PATH, study.study_name, f"trial_{study.best_trial.number}")
    return workdir

def get_best_trial_workdir(study):
    workdir=os.path.join(CHECKPOINT_PATH, study.study_name, f"trial_{study.best_trial.number}")
    return workdir

def remove_bad_trials(study):
    # create new study
    new_study_name = study.study_name + "_trimmed"
    db_path = os.path.join(CHECKPOINT_PATH, new_study_name, "optuna_hparam_search.db")
    if not os.path.exists(os.path.join(CHECKPOINT_PATH, new_study_name)):
        os.makedirs(os.path.join(CHECKPOINT_PATH, new_study_name))

    new_study = optuna.create_study(
        study_name=new_study_name,
        storage=f'sqlite:///{db_path}', # generates a new db if it doesn't exist
        direction=study.direction,
        pruner=study.pruner,
        load_if_exists=False, 
    )

    # load the good trials 
    trials_to_keep = []
    for t in study.get_trials():
        if (t.state == optuna.trial.TrialState.COMPLETE) and (t.values[0] < 1):
            intermed_values_not_huge = True
            for iv in t.intermediate_values.values():
                if iv >= 2:
                    intermed_values_not_huge = False
                    break

            if intermed_values_not_huge:
                trials_to_keep.append(t)

    # trials_to_keep = [t for t in study.get_trials() if (
    #     (t.state == optuna.trial.TrialState.COMPLETE) # trial did not crash
    #     and (t.values[0] < 1) # final val loss is within ok range
    #     and (iv < 10 for iv in t.intermediate_values.values()) # intermed values are not crazy high 
    # )]
    new_study.add_trials(trials_to_keep)

    return new_study
    
