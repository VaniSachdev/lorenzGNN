################################################################################
# This file contains helper functions for tuning hyperparameters.             #
################################################################################

# imports
from spektral.models import GCN
from spektral.data import MixedLoader
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.losses import Loss
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
from datetime import datetime
from plotters import plot_with_predictions
from models import GCN3
import os
import json

DEFAULT_EPOCHS = 5


def train_model(train,
                val,
                model,
                optimizer=Adam,
                learning_rate=0.001,
                loss='mean_squared_error',
                epochs=DEFAULT_EPOCHS,
                verbose=1,
                early_stopping=True,
                early_stopping_patience=2,
                early_stopping_start_from_epoch=3,
                eval_val=True,
                tensorboard=False,
                save_model=True):
    """ Args:
            model: tensorflow model (not yet compiled)
    """
    # type checks
    assert isinstance(optimizer, str) or isinstance(optimizer(), Optimizer)
    assert isinstance(loss, str) or isinstance(loss(), Loss)

    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    if eval_val:
        assert val is not None
        val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    # compile model
    model.compile(optimizer=optimizer(learning_rate), loss=loss)

    # set up callbacks, logging, etc
    fname = "{}-{}".format(model.name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = []

    if early_stopping:
        callback_early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=early_stopping_patience,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=early_stopping_start_from_epoch)
        callbacks.append(callback_early_stop)
    if tensorboard:
        log_dir = os.path.join('logs/fit', fname)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

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

    if save_model:
        # save the model
        fpath = os.path.join('saved_models', fname)
        model.save(fpath,
                   overwrite=False,
                   include_optimizer=True,
                   save_format='tf',
                   save_traces=True)

        # add the model name and parameters to the index for easier lookup
        model_params = model.get_config()
        train_params = {
            'train_size': len(train),
            'epochs': epochs,
            'optimizer':
            optimizer if isinstance(optimizer, str) else optimizer().name,
            'learning_rate': learning_rate,
            'loss': loss if isinstance(loss, str) else loss().name,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_start_from_epoch': early_stopping_start_from_epoch
        }
        data_params = train.get_config()
        config = {
            "name": model.name,
            "fname": fname,
            "model_params": model_params,
            "train_params": train_params,
            "data_params": data_params
        }

        # append to config index
        with open("configs/index.json", "r") as indexfile:
            index = json.load(indexfile)
            index.append(config)

        with open("configs/index.json", "w") as indexfile:
            json.dump(index, indexfile, indent=4)

    return model, history, fname


def make_train_GCN3(train,
                    val=None,
                    channels_0=2048,
                    channels_1=32,
                    activation="relu",
                    use_bias=False,
                    dropout_rate=0.5,
                    l2_reg=2.5e-4,
                    optimizer=Adam,
                    learning_rate=0.001,
                    loss='mean_squared_error',
                    epochs=DEFAULT_EPOCHS,
                    verbose=1,
                    early_stopping=True,
                    early_stopping_patience=2,
                    early_stopping_start_from_epoch=3,
                    eval_val=True,
                    tensorboard=False):
    # create and train model
    GCN_model = GCN3(
        n_labels=1,
        channels_0=channels_0,  # i.e. n_hidden layers in each GCNConv layer
        channels_1=channels_1,  # i.e. n_hidden layers in each GCNConv layer
        activation=activation,
        output_activation=None,  # we want regression, i.e. a linear function
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg)

    GCN_model, history, fname = train_model(
        train,
        val,
        GCN_model,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        verbose=verbose,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_start_from_epoch=early_stopping_start_from_epoch,
        eval_val=eval_val,
        tensorboard=tensorboard)

    return GCN_model, history, fname


def make_train_GCN(train,
                   val=None,
                   channels=32,
                   activation="relu",
                   use_bias=False,
                   dropout_rate=0.5,
                   l2_reg=2.5e-4,
                   optimizer=Adam,
                   learning_rate=0.001,
                   loss='mean_squared_error',
                   epochs=DEFAULT_EPOCHS,
                   verbose=1,
                   early_stopping=True,
                   early_stopping_patience=2,
                   early_stopping_start_from_epoch=3,
                   eval_val=True,
                   tensorboard=False):
    # create and train model
    GCN_model = GCN(
        n_labels=1,
        channels=channels,  # i.e. n_hidden layers in each GCNConv layer
        activation=activation,
        output_activation=None,  # we want regression, i.e. a linear function
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg)

    GCN_model, history, fname = train_model(
        train,
        val,
        GCN_model,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        epochs=epochs,
        verbose=verbose,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_start_from_epoch=early_stopping_start_from_epoch,
        eval_val=eval_val,
        tensorboard=tensorboard)

    return GCN_model, history, fname


def plot_model_results(train,
                       val,
                       model,
                       history=None,
                       fname=None,
                       epochs=DEFAULT_EPOCHS,
                       ylim=None,
                       save=True):
    # plot training MSE (if model needed training)
    if history is not None:
        fig_train_loss, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(history.history['loss'], label='loss')
        ax.text(x=epochs,
                y=ax.get_ylim()[1],
                s="final train loss: {:.2f}".format(
                    history.history['loss'][-1]))
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='validation loss')
            ax.text(x=epochs,
                    y=0.5 * ax.get_ylim()[1],
                    s="final val loss: {:.2f}".format(
                        history.history['val_loss'][-1]))
        ax.set_ylabel('mean squared error')
        ax.set_xlabel('epochs')
        fig_train_loss.suptitle('model MSE over training epochs')
        if ylim is not None:
            ax.set_ylim(ylim)
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
        model_name=model.name + ' (train)')

    # plot val predictions
    fig_val_pred = plot_with_predictions(
        model=model,
        graph_dataset=val,
        Loader=MixedLoader,
        batch_size=32,
        #  node=0,
        model_name=model.name + ' (val)')

    # save outputs
    if save:
        if fname is None:
            print('no file name, could not save')
        else:
            fdir = os.path.join('figs', fname)
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            fig_train_loss.savefig(os.path.join(fdir, 'fig_train_loss'))
            fig_train_pred.savefig(os.path.join(fdir, 'fig_train_pred'))
            fig_val_pred.savefig(os.path.join(fdir, 'fig_val_pred'))

    return fig_train_loss, fig_train_pred, fig_val_pred


def tune_params(train,
                val,
                param_dict,
                model_class=GCN,
                epochs=20,
                verbose=1,
                early_stopping=True,
                early_stopping_patience=2,
                early_stopping_start_from_epoch=3,
                eval_val=True):
    # set default params if they weren't passed in
    if 'channels' not in param_dict.keys():
        param_dict['channels'] = [32]
    if 'activation' not in param_dict.keys():
        param_dict['activation'] = ['relu']
    if 'use_bias' not in param_dict.keys():
        param_dict['use_bias'] = [False]
    if 'dropout_rate' not in param_dict.keys():
        param_dict['dropout_rate'] = [0.5]
    if 'l2_reg' not in param_dict.keys():
        param_dict['l2_reg'] = [2.5e-4]
    if 'optimizer' not in param_dict.keys():
        param_dict['optimizer'] = [Adam]
    if 'learning_rate' not in param_dict.keys():
        param_dict['learning_rate'] = 0.001
    if 'loss' not in param_dict.keys():
        param_dict['loss'] = ["mean_squared_error"]

    results = {}
    # dictionary where value = param dictionary and key = (model, history) tuple

    for param in list(ParameterGrid(param_dict)):
        print('running with params', param)
        model, history, fname = make_train_GCN(
            train,
            val,
            channels=param['channels'],
            activation=param['activation'],
            use_bias=param['use_bias'],
            dropout_rate=param['dropout_rate'],
            l2_reg=param['l2_reg'],
            optimizer=param['optimizer'],
            learning_rate=param['learning_rate'],
            loss=param['loss'],
            epochs=epochs,
            verbose=verbose,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_start_from_epoch=early_stopping_start_from_epoch,
            eval_val=eval_val)

        results[param] = (model, history)

    return results