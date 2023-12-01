import ml_collections

def get_config():
    """Get the hyperparameter configuration for the GraphNetwork model."""
    config = ml_collections.ConfigDict()

    # Data params. 
    config.n_samples=1_000
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

    # Optimizer.
    config.optimizer = 'adam'
    config.learning_rate = 1e-3

    # Training hyperparameters.
    config.batch_size = 3
    config.epochs = 200
    config.log_every_epochs = 1
    config.eval_every_epochs = 10
    config.checkpoint_every_epochs = 10
    # config.num_train_steps = 100_000 # TODO is this different from epochs?
    # config.log_every_steps = 2
    # config.eval_every_steps = 1
    # config.checkpoint_every_steps = 2
    config.add_virtual_node = True # TODO what is this? 
    config.add_undirected_edges = True # TODO what is this? 
    config.add_self_loops = True # TODO what is this? 

    # GNN hyperparameters.
    config.model = 'MLPBlock'
    config.dropout_rate = 0.1
    config.skip_connections = False # This was throwing a broadcast error in add_graphs_tuples_nodes when this was set to True
    config.layer_norm = False # TODO perhaps we want to turn on later
    config.edge_features = (4, 8) # the last feature size will be the number of features that the graph predicts
    config.node_features = (32, 2)
    config.global_features = None

    return config
