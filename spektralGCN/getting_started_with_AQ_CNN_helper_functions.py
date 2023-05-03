import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from matplotlib import dates as d
from matplotlib import pylab
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from keras.layers import Activation, Conv1D, Dense, Flatten, MaxPooling1D
from keras.metrics import MeanSquaredLogarithmicError
from keras.models import Sequential
from keras.regularizers import l2

from sklearn import preprocessing
from sklearn.metrics import r2_score

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

cp = sns.color_palette("coolwarm", 12) + sns.color_palette("coolwarm_r", 12)
my_cmap = colors.ListedColormap(sns.color_palette(cp).as_hex())

####################### Some quick helper functions #####################


'''Split the numpy data into a format a 1D CNN or LSTM can use'''
def split_data(data_np, n_steps=24):
    X, y = list(), list() # making empty lists to store our X and y
    for i in range(len(data_np)):
        end_ix = i + n_steps 
        if end_ix > len(data_np):
            break
        # this will reshape the data for input into the CNN
        data_x, data_y = data_np[i:end_ix, :-1], data_np[end_ix-1, -1]
        X.append(data_x)
        y.append(data_y)
    return np.array(X), np.array(y)

#####################################################################  
'''Unnormalize/scale the data'''
def invTransform(scaler, X, y):
    X_y = np.zeros([X.shape[0],X.shape[-1]])
    X_y = np.column_stack((X_y, y))
    X_y_inv = scaler.inverse_transform(X_y)
    return X_y_inv[:,-1]

#####################################################################  
'''Plot training and validation loss and metrics'''
def plot_training_validation_metrics(history_from_fit): 
    keys = list(history_from_fit.history.keys())
    num_plots = int(len(keys)/2)

    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history_from_fit.epoch, history_from_fit.history[keys[0]], label="Train loss")
    ax[0].plot(history_from_fit.epoch, history_from_fit.history[keys[num_plots]], label="Validation loss")
    ax[1].set_title('metrics')
    for i in range(1,num_plots*2):
      if i != num_plots: ax[1].plot(history_from_fit.epoch, history_from_fit.history[keys[i]], label=keys[i])
    ax[0].legend()
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_yscale('log')
    plt.show()
    
#####################################################################    
'''Calculate likelihood of y_test values having been in y_train'''    
def some_weird_stats(y_train, y_test, y_predicted, bin_step_size=5):
  histogram_bin_for_test_data = np.full(len(y_test), np.nan)
  bin_upper_bound_for_test_data = np.full(len(y_test), np.nan)

  # makes a histogram of y_training values
  bin_range = np.arange(0, max(y_train.max(),y_test.max())+bin_step_size, bin_step_size)
  out, bins  = pd.cut(y_train, bins=bin_range, include_lowest=True, right=False, retbins=True)
  sum_val = sum(out.value_counts())

  # makes a time series of what histogram bin a given y_test value falls in
  for i in range(len(y_test)):
    for b in range(1,len(bins)):
      if y_test[i] < bins[b]: 
        # stores the normalized count for a given bin (a measure of statistical likelihood)
        histogram_bin_for_test_data[i] = out.value_counts()[b-1]/sum_val
        # also stores the upper bound for that bin (ie. was it 0 to 5 ppb, 5 to 10 ppb, etc.)
        bin_upper_bound_for_test_data[i] = bins[b]
        break

  # make a new data frame to also include model error and bias for each point in the test set
  test_stats = pd.DataFrame() 
  test_stats['hist_bin_count'] = histogram_bin_for_test_data
  test_stats['ozone_bias'] = (y_predicted - y_test)
  test_stats['ozone_error'] = abs(y_predicted - y_test)
  test_stats['bin_upper_bound'] = bin_upper_bound_for_test_data 


  # returning the mean values for each statistical bin
  return test_stats.groupby(['hist_bin_count']).mean()

#####################################################################  
'''Plot y_pred bias and error by likelihood of y_test being in y_train'''
def graph_bias_and_error_by_statistics(y_train, y_test, y_pred, df, label):
  max = df['Ozone'].max()
  min = df['Ozone'].min()

  y_pred = y_pred*(max-min)+min
  y_pred = y_pred*1000

  y_train = y_train*(max-min)+min
  y_train = y_train*1000

  y_test = y_test*(max-min)+min
  y_test = y_test*1000

  control_y_pred = some_weird_stats(y_train, y_test, y_pred)

  f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))
  f.suptitle('Model performance in test period vs representation of test data in training set', fontsize=30)

  ax1.vlines(0, 0, control_y_pred.index.max(), color='grey', alpha=0.8, linestyle='--')
  ax2.vlines(0, 0, control_y_pred.index.max(), color='grey', alpha=0.8, linestyle='--')
  ax1.plot(control_y_pred['ozone_bias'], control_y_pred.index, c='g',linewidth=6, label=label)
  ax2.plot(control_y_pred['ozone_error'], control_y_pred.index, c='g',linewidth=6)

  # this label could be better
  ax1.set_ylabel("Fraction of training data similar to test value", fontsize=20)
  ax1.set_xlabel("mean bias (ppb)", fontsize=25)
  ax2.set_xlabel("mean error (ppb)", fontsize=25)

  ax1.legend(fontsize=25)

  return plt.show()