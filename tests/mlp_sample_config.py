import ml_collections

def get_config():
    """Get the hyperparameter configuration for the GraphNetwork model."""
    config = ml_collections.ConfigDict()

    # Data params. 
    config.n_samples=10
    config.input_steps=3
    config.output_delay=0
    config.output_steps=2
    config.timestep_duration=1
    config.sample_buffer=1
    config.time_resolution=100
    config.init_buffer_samples=0
    config.train_pct=0.2
    config.val_pct=0.4
    config.test_pct=0.4
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
    config.epochs = 4
    config.log_every_epochs = 1
    config.eval_every_epochs = 1
    config.checkpoint_every_epochs = 1
    # config.num_train_steps = 100_000 # TODO is this different from epochs?
    # config.log_every_steps = 2
    # config.eval_every_steps = 1
    # config.checkpoint_every_steps = 2
    config.add_virtual_node = True # TODO what is this? 
    config.add_undirected_edges = True # TODO what is this? 
    config.add_self_loops = True # TODO what is this? 

    # GNN hyperparameters.
    config.model = 'MLPBlock'
    #   config.message_passing_steps = 5
    #   config.latent_size = 256
    config.dropout_rate = 0.1
    #   config.num_mlp_layers = 1
    #   config.num_classes = 128
    #   config.use_edge_model = True
    config.skip_connections = False # This was throwing a broadcast error in add_graphs_tuples_nodes when this was set to True
    config.layer_norm = False # TODO perhaps we want to turn on later
    config.edge_features = (4, 8) # the last feature size will be the number of features that the graph predicts
    config.node_features = (32, 2)
    config.global_features = None

    return config
