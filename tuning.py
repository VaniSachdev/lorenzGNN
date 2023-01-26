################################################################################
# This file contains helper functions for tuning hyperparameters.             #
################################################################################

# imports
from spektral.models import GCN
from spektral.data import MixedLoader
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from plotters import plot_with_predictions

DEFAULT_EPOCHS = 5


def train_model(train,
                val,
                model,
                epochs=DEFAULT_EPOCHS,
                verbose=1,
                early_stopping=True,
                eval_val=True):
    """ Args:
            model: tensorflow model (must be already compiled)
    """
    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    if early_stopping:
        callback_early_stop = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=2,
                                            verbose=0,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=True,
                                            start_from_epoch=3)
        callbacks = [callback_early_stop]
    else:
        callbacks = []

    if eval_val:
        history = model.fit(train_loader.load(),
                            steps_per_epoch=train_loader.steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_loader.load(),
                            validation_steps=val_loader.steps_per_epoch,
                            validation_freq=1,
                            shuffle=False,
                            verbose=verbose)
    else:
        history = model.fit(train_loader.load(),
                            steps_per_epoch=train_loader.steps_per_epoch,
                            epochs=epochs,
                            callbacks=callbacks,
                            shuffle=False,
                            verbose=verbose)
    return model, history


def make_train_GCN(
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
        early_stopping=True,
        eval_val=True):
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

    GCN_model, history = train_model(train,
                                     val,
                                     GCN_model,
                                     epochs=epochs,
                                     verbose=verbose,
                                     early_stopping=early_stopping,
                                     eval_val=eval_val)

    return GCN_model, history


# TODO: clean up these helper functions, not all are being used
def make_train_naive_constant(train, val=None, eval_val=True):
    """ 
        make a naive model that uses the last data point in the input window as its prediction. 
    """
    # TODO: modularize these make_train_[mode] functions
    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    # create and train model
    naive_constant_model = NotImplementedError


def make_train_naive_linear(train, val, eval_val=True):
    """ 
        make a naive model that uses the last data point in the input window as the input to a linear regression model. 
    """
    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    # create and train model
    naive_linear_model = NotImplementedError


def plot_model_results(train,
                       val,
                       model,
                       history=None,
                       model_name='',
                       epochs=DEFAULT_EPOCHS):
    # plot training MSE (if model needed training)
    if history is not None:
        fig_train_loss, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(history.history['loss'], label='loss')
        ax.text(x=epochs,
                y=history.history['loss'][-1],
                s="final train loss: {:.2f}".format(
                    history.history['loss'][-1]))
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='validation loss')
            ax.text(x=epochs,
                    y=history.history['val_loss'][-1],
                    s="final val loss: {:.2f}".format(
                        history.history['val_loss'][-1]))
        ax.set_ylabel('mean squared error')
        ax.set_xlabel('epochs')
        fig_train_loss.suptitle('model MSE over training epochs')
        ax.set_ylim(0, history.history['loss'][0] * 1.1)
        ax.legend()
    else:
        fig_train_loss = None

    # plot train predictions
    fig_train_pred = plot_with_predictions(
        model=model,
        graph_dataset=train,
        Loader=MixedLoader,
        batch_size=32,
        #    node=0,
        model_name=model_name + ' (train)')

    # plot val predictions
    fig_val_pred = plot_with_predictions(
        model=model,
        graph_dataset=val,
        Loader=MixedLoader,
        batch_size=32,
        #  node=0,
        model_name=model_name + ' (val)')

    return fig_train_loss, fig_train_pred, fig_val_pred