import jax.random
import optax
from flax.training import train_state  # Simple train state for the common case with a single Optax optimizer.

from utils.jraph_data import get_lorenz_graph_tuples


def get_sample_data(seed=42):
    n_samples=10
    input_steps=3
    output_steps=2
    train_pct=.2
    val_pct=0.4
    test_pct=0.4
    K=36
    F=8
    c=10
    b=10
    h=1

    sample_dataset = get_lorenz_graph_tuples(n_samples=n_samples,
                        input_steps=3,
                        output_delay=0,
                        output_steps=2,
                        timestep_duration=1,
                        sample_buffer=1,
                        time_resolution=100,
                        init_buffer_samples=0,
                        train_pct=train_pct,
                        val_pct=val_pct,
                        test_pct=test_pct,
                        K=36,
                        F=8,
                        c=10,
                        b=10,
                        h=1,
                        seed=seed,
                        normalize=False)

    return sample_dataset, {"n_samples": n_samples,
                            "input_steps": input_steps,
                            "output_steps": output_steps,
                            "train_pct": train_pct,
                            "val_pct": val_pct,
                            "test_pct": test_pct,
                            "K": K,
                            "F": F,
                            "c": c,
                            "b": b,
                            "h": h,}

def state_setup_helper(model):
    """ Helper function to set up the state, which also contains the params 
        and optimizer. 
    """
    sample_dataset, data_params = get_sample_data()
    sample_input_window = sample_dataset['train']['inputs'][0]

    # we need to set up a state so that we can do the loss computations 
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    params = jax.jit(model.init)(init_rng, sample_input_window)

    # set up optimizer (needed for the state even if we aren't training)
    learning_rate = 0.001  # default learning rate for adam in keras
    tx = optax.adam(learning_rate=learning_rate)

    # set up state object, which helps us keep track of the model, params, and optimizer
    state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=tx)

    return state 


