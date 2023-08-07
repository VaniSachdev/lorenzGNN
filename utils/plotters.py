################################################################################
# This file contains helper functions for plotting things.                     #
################################################################################

# imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_true_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(y_true, y_pred, c='crimson')

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.axis('equal')
    ax.axis('square')

    # ax.plot(y_pred - y_true, marker='o',linestyle='')

    return fig, ax


def plot_with_predictions(model,
                          graph_dataset,
                          Loader,
                          batch_size=32,
                          nodes=[0, 10, 20],
                          model_name=''):
    """ 
        Args:
            model: a tensorflow/keras/spektral model
            graph_dataset: a spektral Dataset object
            node (int): node for which data will be plotted
            model_name (str): (optional) model name to be displayed on the plot
    """
    # generate predictions
    loader = Loader(dataset=graph_dataset, batch_size=batch_size, shuffle=False)
    predictions = model.predict(loader.load(), steps=loader.steps_per_epoch)

    # set up plot
    n_nodes = len(nodes)
    fig = plt.figure(tight_layout=True, figsize=(25, 5 * n_nodes))
    gs = GridSpec(2 * n_nodes, 2, figure=fig)
    title = "time series forecasting"
    if model_name != '':
        title += " for " + model_name
    fig.suptitle(title, size=28)

    # add plots for each node
    for i, node in enumerate(nodes):
        # create subplots
        ax_X1 = fig.add_subplot(gs[i * 2, 0])
        ax_X2 = fig.add_subplot(gs[i * 2 + 1, 0])
        ax_performance = fig.add_subplot(gs[i * 2:i * 2 + 2, 1])

        # format subplots
        ax_X1.set_title(
            "node {} \nX1 (atmospheric variable) time series".format(node),
            size=20)
        ax_X2.set_title(
            "node {} \nX2 (oceanic variable) time series".format(node), size=20)
        # ax_zoomtimeseries.set_title("time series zoomed into the last week",
        #                             size=20)
        ax_X1.set_xlabel('time (days)', size=16)
        ax_X2.set_xlabel('time (days)', size=16)
        ax_performance.set_xlabel('True Values')
        ax_performance.set_ylabel('Predictions')
        ax_performance.axis('equal')
        ax_performance.axis('square')
        # ax_performance.set_aspect('equal', 'box')
        ax_performance.set_ylim((-2, 2))
        ax_performance.set_xlim((-2, 2))

        if graph_dataset.predict_from == "X1X2_window":
            for j, g in enumerate(graph_dataset):
                # plot input data
                ax_X1.plot(
                    g.t_X,
                    g.x[node][:graph_dataset.input_steps],  # only plotting X1
                    label='inputs',
                    c='purple',
                    alpha=0.75)
                ax_X2.plot(
                    g.t_X,
                    g.x[node][graph_dataset.input_steps:],  # plots X2
                    label='inputs',
                    c='purple',
                    alpha=0.75)
                # ax_zoomtimeseries.plot(
                #     g.t_X[-7 * DEFAULT_TIME_RESOLUTION:],
                #     g.x[node][:graph_dataset.input_steps],  # only plotting X1
                #     label='inputs',
                #     c='purple',
                #     alpha=0.75)

                #plot true output
                ax_X1.scatter(g.t_Y,
                              g.y[node][:graph_dataset.output_steps],
                              label='outputs',
                              s=30,
                              c='green')
                # TODO: fix, this is not plotting a week's duration properly
                # ax_zoomtimeseries.scatter(g.t_Y[-7 * DEFAULT_TIME_RESOLUTION:],
                #                           g.y[node][:graph_dataset.output_steps]
                #                           [-7 * DEFAULT_TIME_RESOLUTION:],
                #                           label='output',
                #                           s=30,
                #                           c='green')

                # plot predictions
                pred = predictions[j][node]
                ax_X1.scatter(g.t_Y, pred, label='prediction', s=30, c='red')
                # ax_zoomtimeseries.scatter(g.t_Y[-7 * DEFAULT_TIME_RESOLUTION:],
                #                           pred,
                #                           label='prediction',
                #                           s=30,
                #                           c='red')

                # plot performance
                ax_performance.scatter(g.y[node][:graph_dataset.output_steps],
                                       pred,
                                       c='blue')
                ax_performance.plot(ax_performance.get_xlim(),
                                    ax_performance.get_xlim())
                ax_performance.plot([0, 0], ax_performance.get_xlim())
                ax_performance.plot(ax_performance.get_xlim(), [0, 0])

        elif graph_dataset.predict_from == "X2":
            for i, g in enumerate(graph_dataset):
                # plot input data
                ax_X2.scatter(g.t, g.x[node], label='inputs', c='purple', s=30)

                # plot true output
                ax_X1.scatter(g.t, g.y[node], label='outputs', s=30, c='green')

                # plot predictions
                pred = predictions[i][node]
                ax_X1.scatter(g.t, pred, label='prediction', s=30, c='red')

                # plot performance
                ax_performance.scatter(g.y[node], pred, c='blue')
                ax_performance.plot(ax_performance.get_xlim(),
                                    ax_performance.get_xlim())
                ax_performance.plot([0, 0], ax_performance.get_xlim())
                ax_performance.plot(ax_performance.get_xlim(), [0, 0])

        ax_X2.set_xlim(ax_X1.get_xlim())
    return fig


def plot_data(train,
              val=None,
              test=None,
              node=0,
              fig=None,
              ax0=None,
              ax1=None,
              alpha=0.2):
    """ Plot the time series data for a single node in the graph. 
    
        Returns: 
            fig, (ax0, ax1)
    """
    colors = ["darkorange", "purple", "darkcyan"]

    # set up plot
    if fig is None or ax0 is None or ax1 is None:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 8))
        fig.suptitle("sampled time series after reshaping", size=28)
        ax0.set_title("X1 (i.e. atmospheric variable) for node {}".format(node),
                      size=20)
        ax1.set_title("X2 (i.e. oceanic variable) for node {}".format(node),
                      size=20)
        plt.xlabel('time (days)', size=16)

    # plot train, val, and test data
    print('plotting train')
    fig, (ax0, ax1) = train.plot(node,
                                 fig,
                                 ax0,
                                 ax1,
                                 data_type='train',
                                 color=colors[0],
                                 alpha=alpha)
    if val is None:
        print('no val data to plot')
    else:
        print('plotting val')
        fig, (ax0, ax1) = val.plot(node,
                                   fig,
                                   ax0,
                                   ax1,
                                   data_type='val',
                                   color=colors[1],
                                   alpha=alpha)
    if test is None:
        print('no training data to plot')
    else:
        print('plotting test')
        fig, (ax0, ax1) = test.plot(node,
                                    fig,
                                    ax0,
                                    ax1,
                                    data_type='test',
                                    color=colors[2],
                                    alpha=alpha)

    # ax0.set_xlim(train[0].t_X[0], test[-1].t_Y[-1])
    # ax1.set_xlim(train[0].t_X[0], test[-1].t_Y[-1])
    # ax0.set_xlim(left=0)
    ax1.set_xlim(ax0.get_xlim())

    print('editing legend')
    # create legend
    ax0.legend(loc="upper left")
    # TODO: fix legend labels
    ax0.legend(handles=ax0.get_legend().legendHandles[0:6])
    # leg = ax0.get_legend()
    # [
    #     leg.legendHandles[i].set_color(colors[i // 2])
    #     for i in range(len(leg.legendHandles))
    # ]

    return fig, (ax0, ax1)