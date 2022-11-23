################################################################################
# This file contains helper functions for testing hyperparameters.             #
################################################################################

# imports
from spektral.models import GCN
from spektral.data import MixedLoader

import matplotlib.pyplot as plt
from plotters import plot_with_predictions

DEFAULT_EPOCHS = 5


def make_train_model(
    train,
    val,
    channels=32,
    activation="relu",
    use_bias=False,
    dropout_rate=0,
    l2_reg=0,
    optimizer='adam',
    loss='mean_squared_error',
    epochs=DEFAULT_EPOCHS,  # TODO: add early stopping 
    verbose=1,
    eval_val=True,
):
    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    # create and train model
    GCN_model = GCN(
        n_labels=1,
        channels=channels,  # i.e. n_hidden layers in each GCNConv layer
        activation=activation,
        output_activation=None,  # we want regression, i.e. a linear function
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg)

    GCN_model.compile(optimizer=optimizer, loss=loss)

    if eval_val:
        history = GCN_model.fit(train_loader.load(),
                                steps_per_epoch=train_loader.steps_per_epoch,
                                epochs=epochs,
                                validation_data=val_loader.load(),
                                validation_freq=1,
                                shuffle=False,
                                verbose=verbose)
    else:
        history = GCN_model.fit(train_loader.load(),
                                steps_per_epoch=train_loader.steps_per_epoch,
                                epochs=epochs,
                                shuffle=False,
                                verbose=verbose)
    return GCN_model, history


def plot_model_results(train, val, GCN_model, history, epochs=DEFAULT_EPOCHS):
    # plot training MSE
    fig_train_loss, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(history.history['loss'])
    ax.set_ylabel('mean squared error')
    ax.set_xlabel('epochs')
    fig_train_loss.suptitle('model MSE over training epochs')
    ax.text(x=0.8 * epochs,
            y=1.2 * history.history['loss'][-1],
            s="final loss: {:.2f}".format(history.history['loss'][-1]))
    ax.set_ylim(0, history.history['loss'][0] * 1.1)
    ax.legend()

    # plot train predictions
    fig_train_pred, (ax0,
                     ax1) = plot_with_predictions(model=GCN_model,
                                                  graph_dataset=train,
                                                  Loader=MixedLoader,
                                                  batch_size=32,
                                                  node=0,
                                                  model_name='GCN OOTB train')

    # plot val predictions
    fig_val_pred, (ax0, ax1) = plot_with_predictions(model=GCN_model,
                                                     graph_dataset=val,
                                                     Loader=MixedLoader,
                                                     batch_size=32,
                                                     node=0,
                                                     model_name='GCN OOTB val')

    # # plot predictions against true value
    # fig_true_vs_pred, ax = plot_true_vs_pred(y_true, y_pred)

    plt.tight_layout()
    return fig_train_loss, fig_train_pred, fig_val_pred