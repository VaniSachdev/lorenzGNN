DEFAULT_EPOCHS = 5


def train_model(train,
                val,
                model,
                optimizer=Adam,
                learning_rate=0.001,
                loss='mean_squared_error',
                epochs=DEFAULT_EPOCHS,
                batch_size=32,
                verbose=1,
                early_stopping=True,
                early_stopping_patience=2,
                early_stopping_start_from_epoch=3,
                eval_val=True,
                tensorboard=False,
                save_model=True):
    """ Args:
            model: sonnet Module
    """
    print('in train_model')

    # type checks
    assert isinstance(optimizer, str) or isinstance(optimizer(), Optimizer)
    assert isinstance(loss, str) or isinstance(loss(), Loss)

    # prepare data
    train_loader = MixedLoader(dataset=train, batch_size=32, shuffle=False)
    if eval_val:
        assert val is not None
        val_loader = MixedLoader(dataset=val, batch_size=32, shuffle=False)

    # compile model
    print('compile model')

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

    print('fit model')

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
        print('saving model')
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
