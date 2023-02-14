################################################################################
# This file contains helper functions for running sanity checks.               #
################################################################################

# imports
import matplotlib.pyplot as plt

from tuning import make_train_GCN, plot_model_results

DEFAULT_EPOCHS = 5


def check_init_error():
    """ sanity check initial error matches expected calculations. """
    pass


def check_regularization(train,
                         val,
                         activation="relu",
                         epochs=10,
                         loss='mean_squared_error',
                         verbose=1,
                         tensorboard=False):
    """ sanity check that increasing regularization causes increasing loss. """
    # create models with and without regularization
    model_noreg, history_noreg, fname_noreg = make_train_GCN(
        train,
        val,
        activation=activation,
        epochs=epochs,
        eval_val=False,
        early_stopping=False,
        dropout_rate=0,
        l2_reg=0,
        loss=loss,
        verbose=verbose,
        tensorboard=tensorboard)
    model_smreg, history_smreg, fname_smreg = make_train_GCN(
        train,
        val,
        activation=activation,
        epochs=epochs,
        eval_val=False,
        early_stopping=False,
        dropout_rate=0.25,
        l2_reg=5e-4,
        loss=loss,
        verbose=verbose,
        tensorboard=tensorboard)
    model_lgreg, history_lgreg, fname_lgreg = make_train_GCN(
        train,
        val,
        activation=activation,
        epochs=epochs,
        eval_val=False,
        early_stopping=False,
        dropout_rate=0.5,
        l2_reg=5e-3,
        loss=loss,
        verbose=verbose,
        tensorboard=tensorboard)

    # plot losses for visual comparison
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.plot(history_noreg.history['loss'], label='no regularization')
    ax.plot(history_smreg.history['loss'], label='dropout=0.25, l2_reg=5e-4')
    ax.plot(history_lgreg.history['loss'], label='dropout=0.5, l2_reg=5e-3')
    ax.set_ylabel(loss)
    ax.set_xlabel('epochs')
    fig.suptitle('loss with and without regularization')
    ax.set_ylim(bottom=0)  # set ymin, but leave ymax to be the default
    ax.legend()

    # return fig


def check_overfit(train,
                  val,
                  activation="relu",
                  epochs=50,
                  learning_rate=0.001,
                  loss='mean_squared_error',
                  verbose=1,
                  eval_val=False,
                  tensorboard=False):
    """ sanity check that model has flexibility to overfit. using a tiny number of samples and setting regularization to 0, we want to make sure we can actually get to 0 error. 

    """
    # (we also set dropout to 0 since dropout is supposed to help avoid overfitting.)
    model, history = make_train_GCN(train,
                                    val,
                                    activation=activation,
                                    epochs=epochs,
                                    eval_val=eval_val,
                                    early_stopping=False,
                                    dropout_rate=0,
                                    l2_reg=0,
                                    learning_rate=learning_rate,
                                    loss=loss,
                                    verbose=verbose,
                                    tensorboard=tensorboard)

    fig_train_loss, fig_train_pred, fig_val_pred = plot_model_results(
        train, val, model, history, epochs=epochs)

    return fig_train_loss, fig_train_pred, fig_val_pred
