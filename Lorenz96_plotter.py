###############################################
# By Prof. Sarah Kavassalis; retrieved from 
# http://kavassalis.space/s/Lorenz96_plotter.py 
# and placed in this repo for convenient access
###############################################

from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import pylab
import matplotlib.ticker as mtick
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d
import random

colours = {0: 'darkturquoise',
          1 :'darkblue',
          2:'goldenrod',
          3:'red',
          4:'darkred'}
med_colours = {0: 'paleturquoise',
          1 :'mediumblue',
          2:'gold',
          3:'coral',
          4:'firebrick'}
faint_colours = {0: 'azure',
          1 :'lightskyblue',
          2:'palegoldenrod',
          3:'lightcoral',
          4:'indianred'}

params = {'legend.fontsize': 'xx-large',
         # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


def plot_Lorenz96ts(X, t, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median):
    if X.ndim == 3 and global_avg == False:
        for i in range(ensem_size):
            if ensemoff==False and global_avg== False: plt.plot(t, X[i,:,0], label=label_lead+', ens.mem. '+str(i), linewidth=2)
            if ensemoff==False and global_avg== True: 
                for j in range(scenario_size):
                    plt.plot(t, X[j, i,:], label=label_lead+', ens.mem. '+str(i), linewidth=2)
        if mean == True:
            #print("ensemble mean + standard deviation")
            X_ensemble_mean = X.mean(axis=0)
            X_ensemble_std = X.std(axis=0)
            if global_avg == False:
                if ensemoff==False: plt.plot(t, X_ensemble_mean[:,0],colour_mean, linewidth=5, label=label_lead+', ensem. mean')
                elif ensemoff==True: plt.plot(t, X_ensemble_mean[:,0],colour_mean, linewidth=5, label=label_lead+', ensem. mean + standard deviation')
                if ensemoff == True: plt.fill_between(t, X_ensemble_mean[:,0]-X_ensemble_std[:,0], X_ensemble_mean[:,0]+X_ensemble_std[:,0],color=colour_mean, alpha=0.5)
#             else:
#                 for j in range(scenario_size):
#                     plt.plot(t, X_ensemble_mean[j,:],colours[j], linewidth=5, label=label_lead+str(j)+', ens. mean')
#                     if ensemoff == True: plt.fill_between(t, X_ensemble_mean[j,:]-X_ensemble_std[j,:], X_ensemble_mean[j,:]+X_ensemble_std[j,:],color=med_colours[j])
        if median == True:
            #print("ensemble median + 25th/75th + 5th/95th percentiles")
            X_ensemble_median = np.percentile(X, 50, axis=0)
            X_ensemble_25th = np.percentile(X, 25, axis=0)
            X_ensemble_75th = np.percentile(X, 75, axis=0)
            X_ensemble_5th = np.percentile(X, 5, axis=0)
            X_ensemble_95th = np.percentile(X, 95, axis=0)
            if global_avg == False:
                if ensemoff==False: plt.plot(t, X_ensemble_median[:,0],colour_median, linewidth=5, label=label_lead+', ens. median')
                elif ensemoff==True: plt.plot(t, X_ensemble_median[:,0],colour_median, linewidth=5, label=label_lead+', ens. median + 25/75th + 5/95th percentiles')
                
                if ensemoff == True: plt.fill_between(t, X_ensemble_5th[:,0], X_ensemble_95th[:,0], color=colour_median, alpha=0.25)
                if ensemoff == True: plt.fill_between(t, X_ensemble_25th[:,0], X_ensemble_75th[:,0], color=colour_median, alpha=0.5)
#             elif global_avg == True and scenario_size > 0:
#                 for j in range(scenario_size):
#                     plt.plot(t, X_ensemble_median[j,:],colours[j], linewidth=5, label=label_lead+str(j)+', ens. median')
#                     if ensemoff == True: plt.fill_between(t, X_ensemble_5th[j,:], X_ensemble_95th[j,:], color=faint_colours[j])
#                     if ensemoff == True: plt.fill_between(t, X_ensemble_25th[j,:], X_ensemble_75th[j,:], color=med_colours[j])
#             else:
#                 plt.plot(t, X_ensemble_median[:],colour_median, linewidth=5, label=label_lead+str(j)+', ens. median')
#                 if ensemoff == True: plt.fill_between(t, X_ensemble_5th[j,:], X_ensemble_95th[j,:], color=faint_colours[j])
#                 if ensemoff == True: plt.fill_between(t, X_ensemble_25th[j,:], X_ensemble_75th[j,:], color=med_colours[j])
    
    elif X.ndim == 3 and global_avg == True:
        for j in range(scenario_size):
            X_ensemble = X[j]
            for i in range(ensem_size):
                if ensemoff==False: plt.plot(t, X_ensemble[i,:], colours[j], label=label_lead+str(j)+', ens.mem. '+str(i), linewidth=2)

            if mean == True:
                #print("ensemble mean + standard deviation")
                X_ensemble_mean = X_ensemble.mean(axis=0)
                X_ensemble_std = X_ensemble.std(axis=0)
                
                if ensemoff==False: plt.plot(t, X_ensemble_mean[:],colours[j], linewidth=5, label=label_lead+str(j)+', ensem. mean')
                elif ensemoff==True: plt.plot(t, X_ensemble_mean[:],colours[j], linewidth=5, label=label_lead+str(j)+', ensem. mean')
                if ensemoff == True: plt.fill_between(t, X_ensemble_mean[:]-X_ensemble_std[:], X_ensemble_mean[:]+X_ensemble_std[:],color=colours[j], alpha=0.5)

            if median == True:
                #print("ensemble median + 25th/75th + 5th/95th percentiles")
                X_ensemble_median = np.percentile(X_ensemble, 50, axis=0)
                X_ensemble_25th = np.percentile(X_ensemble, 25, axis=0)
                X_ensemble_75th = np.percentile(X_ensemble, 75, axis=0)
                X_ensemble_5th = np.percentile(X_ensemble, 5, axis=0)
                X_ensemble_95th = np.percentile(X_ensemble, 95, axis=0)
               
                if ensemoff==False: plt.plot(t, X_ensemble_median[:],colours[j], linewidth=5, label=label_lead+str(j)+', ens. median')
                elif ensemoff==True: plt.plot(t, X_ensemble_median[:],colours[j], linewidth=5, label=label_lead+str(j)+', ens. median')

                if ensemoff == True: plt.fill_between(t, X_ensemble_5th[:], X_ensemble_95th[:], color=colours[j], alpha=0.25)
                if ensemoff == True: plt.fill_between(t, X_ensemble_25th[:], X_ensemble_75th[:], color=colours[j], alpha=0.5)
                
    
    #################### scenarios + ensemble + NO global average #################################
    elif X.ndim == 4: 
        for j in range(scenario_size):
            X_ensemble = X[j]
            for i in range(ensem_size):
                if ensemoff==False:
                    plt.plot(t, X_ensemble[i,:,0], colours[j], label=label_lead +str(j)+', ens='+str(i), linewidth=2)

            if mean == True:
                #print("ensemble mean + standard deviation")
                X_ensemble_mean = X_ensemble.mean(axis=0)
                X_ensemble_std = X_ensemble.std(axis=0)
                if ensemoff==False: plt.plot(t, X_ensemble_mean[:,0],colours[j], linewidth=5, label=label_lead+str(j)+', ensem. mean')
                elif ensemoff==True: plt.plot(t, X_ensemble_mean[:,0],colours[j], linewidth=5, label=label_lead+str(j)+', ensem. mean')
                if ensemoff == True: plt.fill_between(t, X_ensemble_mean[:,0]-X_ensemble_std[:,0], X_ensemble_mean[:,0]+X_ensemble_std[:,0],color=colours[j], alpha=0.5)


            if median == True:
                #print("ensemble median + 25th/75th + 5th/95th percentiles")
                X_ensemble_median = np.percentile(X_ensemble, 50, axis=0)
                X_ensemble_25th = np.percentile(X_ensemble, 25, axis=0)
                X_ensemble_75th = np.percentile(X_ensemble, 75, axis=0)
                X_ensemble_5th = np.percentile(X_ensemble, 5, axis=0)
                X_ensemble_95th = np.percentile(X_ensemble, 95, axis=0)
                if ensemoff==False: plt.plot(t, X_ensemble_median[:,0],colours[j], linewidth=5, label=label_lead+str(j)+', ens. median')
                elif ensemoff==True: plt.plot(t, X_ensemble_median[:,0],colours[j], linewidth=5, label=label_lead+str(j)+', ens. median')

                if ensemoff == True: plt.fill_between(t, X_ensemble_5th[:,0], X_ensemble_95th[:,0], color=colours[j], alpha=0.25)
                if ensemoff == True: plt.fill_between(t, X_ensemble_25th[:,0], X_ensemble_75th[:,0], color=colours[j], alpha=0.5)
                
    
    ######################################## no ensemble/scenarios OR ensemble + global average ########
    else:
        if global_avg == False:
            plt.plot(t, X[:,0], colour_main, label=label_lead, linewidth=2)
        elif global_avg == True and ensem_size == 0:
            plt.plot(t, X[:], colour_main, label=label_lead, linewidth=2)
        elif global_avg == True and ensem_size > 0:
            if ensemoff==False:
                for i in range(ensem_size):
                    plt.plot(t, X[i,:], label=label_lead+', ens.mem. '+str(i), linewidth=2)
            if mean == True:
                X_ensemble_mean = X.mean(axis=0)
                X_ensemble_std = X.std(axis=0)
                plt.plot(t, X_ensemble_mean[:],colour_median, linewidth=5, label=label_lead+', ens. mean')
                if ensemoff == True: plt.fill_between(t, X_ensemble_mean[:]-X_ensemble_std[:], X_ensemble_mean[:]+X_ensemble_std[:],color=colour_median, alpha=0.5)
            
            if median == True:
                X_ensemble_median = np.percentile(X, 50, axis=0)
                X_ensemble_25th = np.percentile(X, 25, axis=0)
                X_ensemble_75th = np.percentile(X, 75, axis=0)
                X_ensemble_5th = np.percentile(X, 5, axis=0)
                X_ensemble_95th = np.percentile(X, 95, axis=0)
                
                plt.plot(t, X_ensemble_median[:],colour_median, linewidth=5, label=label_lead+', ens. median')
                if ensemoff == True: plt.fill_between(t, X_ensemble_5th[:], X_ensemble_95th[:], color=colour_median, alpha=0.25)
                if ensemoff == True: plt.fill_between(t, X_ensemble_25th[:], X_ensemble_75th[:], color=colour_median, alpha=0.5)


    

def plot_Lorenz96(time_series, lat_plot, X, t, F, K, number_of_days, global_avg=False, extra_ts_plots=False, mean=True, median=False, ensemoff=False, twoD_latplot = False, label_lead='', base_colour=True):
    is_atmos_and_ocean = False
    if X.shape[X.ndim-1] == 2*K:
        X_atmos = np.split(X,2, axis=X.ndim-1)[0]
        X_ocean = np.split(X,2, axis=X.ndim-1)[1]
        print("First component")
        label_lead="First "
        is_atmos_and_ocean = True
        X=X_atmos
        
    if base_colour == True:
        colour_main = 'tab:pink'
        colour_mean = 'maroon'
        colour_median = 'chocolate'
        colour_latplot = 'PiYG_r'
    if base_colour == False:
        colour_main = 'tab:blue'
        colour_mean = 'darkslategrey'
        colour_median = 'steelblue'
        colour_latplot = 'coolwarm'
    
    
    if X.ndim == 4:
        scenario_size = X.shape[0]
        ensem_size = X.shape[1]
    elif X.ndim == 3:
        ensem_size = X.shape[0]
        scenario_size = 0
    else:
        scenario_size = 0
        ensem_size = 0
        
    if global_avg==True and lat_plot == False:
        label_lead='Global avg. X '
        X = X.mean(axis=X.ndim-1)
    elif global_avg==False and lat_plot == False:
        label_lead = '$X_0$'
    elif lat_plot == True:
        label_lead = '$X_k$'
    elif global_avg==True and lat_plot == True:
        Print("No 'lat_plot' option for global average. Assuming global_avg = False.")
        
    if (scenario_size>0 or ensem_size > 0) and twoD_latplot == True:
        print("2D lat plot not compatible with this input")
        
    if scenario_size > 0: 
        label_lead=label_lead+', RCP'
        
    if number_of_days>=1000:
        tplot=t/365
        label_time = '(years)'
    else:
        tplot=t
        label_time = '(days)'

    if len(label_lead) > 20:
        legend_columns = 2
    elif len(label_lead) < 10:
        legend_columns = 4
    else:
        legend_columns = 3
    
    
    # making the time series plots    
    if time_series == True:
        plt.figure(figsize=(20,8))
        plot_Lorenz96ts(X, tplot, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median)
        plt.grid()
        if (scenario_size > 0 or (ensem_size > 0 and ensemoff == False)) and extra_ts_plots == False:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=legend_columns)
        elif (scenario_size == 0 and (ensem_size == 0 or ensemoff == True)):
            plt.legend(loc='best')
        plt.xlabel('time '+label_time,size=22)
        if global_avg == False: plt.ylabel('$X_0$',size=22)
        elif global_avg == True: plt.ylabel('$\overline{X_k}$',size=22)
            
        if scenario_size == 0: plt.title('Lorenz 96 Model, time series, F='+str(F)+", K="+str(K),size=28)
        elif scenario_size > 0: plt.title('Lorenz 96 Model, time series, K='+str(K),size=28)


        
        if extra_ts_plots == True:
            if number_of_days >= 30:
                plt.figure(figsize=(20,4))
                plot_Lorenz96ts(X, t, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median)
                plt.grid()
                plt.xlabel('time (days)',size=22)
                plt.xlim([0, 7])
                if global_avg == False: plt.ylabel('$X_0$',size=22)
                elif global_avg == True: plt.ylabel('$\overline{X_k}$',size=22)
                plt.title('First week of simulation',size=22)
                plt.show()

                plt.figure(figsize=(20,4))
                plot_Lorenz96ts(X, t, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median)
                plt.grid()
                plt.xlabel('time (days)',size=22)
                plt.xlim([number_of_days-7, number_of_days])
                if global_avg == False: plt.ylabel('$X_0$',size=22)
                elif global_avg == True: plt.ylabel('$\overline{X_k}$',size=22)
                plt.title('Final week of simulation',size=22)
                plt.show()

            if number_of_days > 3:
                plt.figure(figsize=(20,2))
                tplot=t*24
                plot_Lorenz96ts(X, tplot, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median)
                plt.grid()
                plt.xlabel('time (hours)',size=22)
                plt.xlim([0, 24])
                if global_avg == False: plt.ylabel('$X_0$',size=22)
                elif global_avg == True: plt.ylabel('$\overline{X_k}$',size=22)
                plt.title('First day of simulation',size=22)
                plt.show()

                plt.figure(figsize=(20,2))
                tplot=tplot-(tplot[-1]-24)
                plot_Lorenz96ts(X, tplot, label_lead, scenario_size, ensem_size, ensemoff, mean, median, global_avg, colour_main, colour_mean, colour_median)
                plt.grid()
                plt.xlabel('time (hours)',size=22)
                plt.xlim([tplot[-1]-24, tplot[-1]])
                if global_avg == False: plt.ylabel('$X_0$',size=22)
                elif global_avg == True: plt.ylabel('$\overline{X_k}$',size=22)
                plt.title('Final day of similation',size=22)
                if (scenario_size > 0 or (ensem_size > 0 and ensemoff == False)):
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=True, shadow=True, ncol=legend_columns)
                plt.show()
    
    ######################################LAT PLOT#####################################################
    if lat_plot == True and twoD_latplot == False and scenario_size == 0:
        lon=np.arange(180,-180,-360/K)
            
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True, figsize=(20,12))
        fig.suptitle('Lorenz 96 Model, longitudinal profiles of $X_k$, time slices, F='+str(F)+", K="+str(K),size=28)
        plt.xlabel('longitude', fontsize=20); plt.rc('xtick', labelsize=16)
        fig.text(0.06, 0.5, '$X_k$', fontsize=26, ha='center', va='center', rotation='vertical')
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax1.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax2.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax3.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax4.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax5.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax6.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax6.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax7.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax7.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax8.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax8.yaxis.set_major_locator(mtick.MaxNLocator(2))
        if X.ndim == 3:
            one_hour = int(len(t)/number_of_days/24)
            six_hours = int(len(t)/number_of_days/12)
            twelve_hours = int(len(t)/number_of_days/2)
            one_day = int(len(t)/number_of_days)
            one_and_a_half_days = int(len(t)/number_of_days*1.5)
            half_sim = len(t)//2
            final_time_stamp = len(t)-1

            for i in range(len(X)):
                X_ensemble = X[i]
                if ensemoff==False:
                    ax1.plot(lon, X_ensemble[0],  linewidth=2)
                    ax2.plot(lon, X_ensemble[one_hour], linewidth=2)
                    ax3.plot(lon, X_ensemble[six_hours], linewidth=2)
                    ax4.plot(lon, X_ensemble[twelve_hours], linewidth=2)
                    ax5.plot(lon, X_ensemble[one_day],linewidth=2)
                    ax6.plot(lon, X_ensemble[one_and_a_half_days], linewidth=2)
                    ax7.plot(lon, X_ensemble[half_sim], linewidth=2)
                    ax8.plot(lon, X_ensemble[final_time_stamp], label='ens.mem. '+str(i), linewidth=2)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6),
          fancybox=True, shadow=True, ncol=legend_columns)
            ############ Mean lat plot ##################
            if mean == True:
                X_ensemble_mean = X.mean(axis=0)
                X_ensemble_std = X.std(axis=0)
                if ensemoff==False: ax1.plot(lon, X_ensemble_mean[0],colour_mean, label='ensemble mean', linewidth=5)
                elif ensemoff==True: ax1.plot(lon, X_ensemble_mean[0],colour_mean, label='ensemble mean + standard deviation', linewidth=5)
                ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, fontsize=20)
                ax2.plot(lon, X_ensemble_mean[one_hour],colour_mean, linewidth=5)
                ax3.plot(lon, X_ensemble_mean[six_hours],colour_mean, linewidth=5)
                ax4.plot(lon, X_ensemble_mean[twelve_hours],colour_mean, linewidth=5)
                ax5.plot(lon, X_ensemble_mean[one_day],colour_mean, linewidth=5)
                ax6.plot(lon, X_ensemble_mean[one_and_a_half_days],colour_mean, linewidth=5)
                ax7.plot(lon, X_ensemble_mean[half_sim],colour_mean, linewidth=5)
                ax8.plot(lon, X_ensemble_mean[final_time_stamp],colour_mean, linewidth=5)
                
                if ensemoff==True:
                    #ax1.text(1, 1, 'ensemble mean + standard deviation', horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes, fontsize=20)
                    ax1.fill_between(lon, X_ensemble_mean[0]-X_ensemble_std[0], X_ensemble_mean[0]+X_ensemble_std[0], color=colour_mean, alpha=0.5)
                    ax2.fill_between(lon, X_ensemble_mean[one_hour]-X_ensemble_std[one_hour], X_ensemble_mean[one_hour]+X_ensemble_std[one_hour], color=colour_mean, alpha=0.5)
                    ax3.fill_between(lon, X_ensemble_mean[six_hours]-X_ensemble_std[six_hours], X_ensemble_mean[six_hours]+X_ensemble_std[six_hours], color=colour_mean, alpha=0.5)
                    ax4.fill_between(lon, X_ensemble_mean[twelve_hours]-X_ensemble_std[twelve_hours], X_ensemble_mean[twelve_hours]+X_ensemble_std[twelve_hours], color=colour_mean, alpha=0.5)
                    ax5.fill_between(lon, X_ensemble_mean[one_day]-X_ensemble_std[one_day], X_ensemble_mean[one_day]+X_ensemble_std[one_day], color=colour_mean, alpha=0.5)
                    ax6.fill_between(lon, X_ensemble_mean[one_and_a_half_days]-X_ensemble_std[one_and_a_half_days], X_ensemble_mean[one_and_a_half_days]+X_ensemble_std[one_and_a_half_days], color=colour_mean, alpha=0.5)
                    ax7.fill_between(lon, X_ensemble_mean[half_sim]-X_ensemble_std[half_sim], X_ensemble_mean[half_sim]+X_ensemble_std[half_sim], color=colour_mean, alpha=0.5)
                    ax8.fill_between(lon, X_ensemble_mean[final_time_stamp]-X_ensemble_std[final_time_stamp], X_ensemble_mean[final_time_stamp]+X_ensemble_std[final_time_stamp], color=colour_mean, alpha=0.5)


            ############ Median lat plot ##################
            if median == True:
                X_ensemble_median = np.percentile(X, 50, axis=0)
                X_ensemble_25th = np.percentile(X, 25, axis=0)
                X_ensemble_75th = np.percentile(X, 75, axis=0)
                X_ensemble_5th = np.percentile(X, 5, axis=0)
                X_ensemble_95th = np.percentile(X, 95, axis=0)
                if ensemoff==False: ax1.plot(lon, X_ensemble_median[0],colour_median, label='ensemble median', linewidth=5)
                elif ensemoff==True: ax1.plot(lon, X_ensemble_median[0],colour_median, label='ensemble median + 25/75th + 5/95th percentiles', linewidth=5)
                ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, fontsize=20)
                ax2.plot(lon, X_ensemble_median[one_hour],colour_median, linewidth=5)
                ax3.plot(lon, X_ensemble_median[six_hours],colour_median, linewidth=5)
                ax4.plot(lon, X_ensemble_median[twelve_hours],colour_median, linewidth=5)
                ax5.plot(lon, X_ensemble_median[one_day],colour_median, linewidth=5)
                ax6.plot(lon, X_ensemble_median[one_and_a_half_days],colour_median, linewidth=5)
                ax7.plot(lon, X_ensemble_median[half_sim],colour_median, linewidth=5)
                ax8.plot(lon, X_ensemble_median[final_time_stamp],colour_median,  linewidth=5)
                
                if ensemoff==True:
                    #ax1.text(1, 1, 'ensemble median + 25/75th + 5/95th percentiles', horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes, fontsize=20)
                    ax1.fill_between(lon, X_ensemble_5th[0], X_ensemble_95th[0], color=colour_median, alpha=0.25)
                    ax1.fill_between(lon, X_ensemble_25th[0], X_ensemble_75th[0], color=colour_median, alpha=0.5)
                    ax2.fill_between(lon, X_ensemble_5th[one_hour], X_ensemble_95th[one_hour], color=colour_median, alpha=0.25)
                    ax2.fill_between(lon, X_ensemble_25th[one_hour], X_ensemble_75th[one_hour], color=colour_median, alpha=0.5)
                    ax3.fill_between(lon, X_ensemble_5th[six_hours], X_ensemble_95th[six_hours], color=colour_median, alpha=0.25)
                    ax3.fill_between(lon, X_ensemble_25th[six_hours], X_ensemble_75th[six_hours], color=colour_median, alpha=0.5)
                    ax4.fill_between(lon, X_ensemble_5th[twelve_hours], X_ensemble_95th[twelve_hours], color=colour_median, alpha=0.25)
                    ax4.fill_between(lon, X_ensemble_25th[twelve_hours], X_ensemble_75th[twelve_hours], color=colour_median, alpha=0.5)
                    ax5.fill_between(lon, X_ensemble_5th[one_day], X_ensemble_95th[one_day], color=colour_median, alpha=0.25)
                    ax5.fill_between(lon, X_ensemble_25th[one_day], X_ensemble_75th[one_day], color=colour_median, alpha=0.5)
                    ax6.fill_between(lon, X_ensemble_5th[one_and_a_half_days], X_ensemble_95th[one_and_a_half_days], color=colour_median, alpha=0.25)
                    ax6.fill_between(lon, X_ensemble_25th[one_and_a_half_days], X_ensemble_75th[one_and_a_half_days], color=colour_median, alpha=0.5)
                    ax7.fill_between(lon, X_ensemble_5th[half_sim], X_ensemble_95th[half_sim], color=colour_median, alpha=0.25)
                    ax7.fill_between(lon, X_ensemble_25th[half_sim], X_ensemble_75th[half_sim], color=colour_median, alpha=0.5)
                    ax8.fill_between(lon, X_ensemble_5th[final_time_stamp], X_ensemble_95th[final_time_stamp], color=colour_median, alpha=0.25)
                    ax8.fill_between(lon, X_ensemble_25th[final_time_stamp], X_ensemble_75th[final_time_stamp], color=colour_median, alpha=0.5)
                    
        else :
            ax1.plot(lon,X[0], colour_main, linewidth=3);
            ax2.plot(lon, X[int(len(t)/number_of_days/24)], colour_main,linewidth=3)
            ax3.plot(lon, X[int(len(t)/number_of_days/12)], colour_main,linewidth=3)
            ax4.plot(lon, X[int(len(t)/number_of_days/2)], colour_main,linewidth=3)
            ax5.plot(lon, X[int(len(t)/number_of_days)], colour_main,linewidth=3)
            ax6.plot(lon, X[int(len(t)/number_of_days*1.5)], colour_main,linewidth=3)
            ax7.plot(lon, X[len(t)//2], colour_main,linewidth=3)
            ax8.plot(lon, X[len(t)-1], colour_main, linewidth=3)
        ax1.text(0.01, .8, 't=0 hours', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
        ax2.text(0.01, .8, 't=1 hour', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
        ax3.text(0.01, .8, 't=6 hours', horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes, fontsize=20)
        ax4.text(0.01, .8, 't=12 hours', horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, fontsize=20)
        ax5.text(0.01, .8, 't=1 day', horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes, fontsize=20)
        ax6.text(0.01, .8, 't=1.5 days', horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes, fontsize=20)
        ax7.text(0.01, .8, 't='+str(number_of_days//2)+' days', horizontalalignment='left', verticalalignment='center', transform=ax7.transAxes, fontsize=20)
        ax8.text(0.01, .8, 't='+str(number_of_days)+' days', horizontalalignment='left', verticalalignment='center', transform=ax8.transAxes, fontsize=20)
    
    
    ##################################### Scenario lat plots ##############################
    elif lat_plot == True and twoD_latplot == False and scenario_size > 0:
        lon=np.arange(180,-180,-360/K)
            
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True, figsize=(20,12))
        fig.suptitle('Lorenz 96 Model, longitudinal profiles of $X_k$, time slices, K='+str(K),size=28)
        plt.xlabel('longitude', fontsize=20); plt.rc('xtick', labelsize=16)
        fig.text(0.06, 0.5, '$X_k$', fontsize=26, ha='center', va='center', rotation='vertical')
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax1.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax2.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax3.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax4.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax5.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax6.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax6.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax7.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax7.yaxis.set_major_locator(mtick.MaxNLocator(2))
        ax8.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax8.yaxis.set_major_locator(mtick.MaxNLocator(2))
        
        one_hour = int(len(t)/number_of_days/24)
        six_hours = int(len(t)/number_of_days/12)
        twelve_hours = int(len(t)/number_of_days/2)
        one_day = int(len(t)/number_of_days)
        one_and_a_half_days = int(len(t)/number_of_days*1.5)
        half_sim = len(t)//2
        final_time_stamp = len(t)-1

        for j in range(scenario_size):
            X_scenario = X[j]
            for i in range(ensem_size):
                X_ensemble = X_scenario[i,:,:]
                if ensemoff==False:
                    ax1.plot(lon, X_ensemble[0],  colours[j], linewidth=2)
                    ax2.plot(lon, X_ensemble[one_hour],colours[j], linewidth=2)
                    ax3.plot(lon, X_ensemble[six_hours],colours[j], linewidth=2)
                    ax4.plot(lon, X_ensemble[twelve_hours],colours[j], linewidth=2)
                    ax5.plot(lon, X_ensemble[one_day],colours[j],linewidth=2)
                    ax6.plot(lon, X_ensemble[one_and_a_half_days],colours[j], linewidth=2)
                    ax7.plot(lon, X_ensemble[half_sim], colours[j], linewidth=2)
                    ax8.plot(lon, X_ensemble[final_time_stamp],colours[j], label=label_lead+str(j)+', ens.mem. '+str(i), linewidth=2)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6),
          fancybox=True, shadow=True, ncol=legend_columns)
            ############ Mean lat plot ##################
            if mean == True:
                X_ensemble_mean = X_scenario.mean(axis=0)
                X_ensemble_std = X_scenario.std(axis=0)
                #if ensemoff==False: ax1.plot(lon, X_ensemble_mean[0],colours[j], label='ensemble mean', linewidth=5)
                ax1.plot(lon, X_ensemble_mean[0],colours[j], linewidth=5)
                #ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, fontsize=20)
                ax2.plot(lon, X_ensemble_mean[one_hour],colours[j], linewidth=5)
                ax3.plot(lon, X_ensemble_mean[six_hours],colours[j], linewidth=5)
                ax4.plot(lon, X_ensemble_mean[twelve_hours],colours[j], linewidth=5)
                ax5.plot(lon, X_ensemble_mean[one_day],colours[j],linewidth=5)
                ax6.plot(lon, X_ensemble_mean[one_and_a_half_days],colours[j], linewidth=5)
                ax7.plot(lon, X_ensemble_mean[half_sim],colours[j], linewidth=5)
                ax8.plot(lon, X_ensemble_mean[final_time_stamp],colours[j], label=label_lead+str(j)+", ens. mean", linewidth=5)

                if ensemoff==True:
                    #ax1.text(1, 1, 'ensemble mean + standard deviation', horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes, fontsize=20)
                    ax1.fill_between(lon, X_ensemble_mean[0]-X_ensemble_std[0], X_ensemble_mean[0]+X_ensemble_std[0])
                    ax2.fill_between(lon, X_ensemble_mean[one_hour]-X_ensemble_std[one_hour], X_ensemble_mean[one_hour]+X_ensemble_std[one_hour])
                    ax3.fill_between(lon, X_ensemble_mean[six_hours]-X_ensemble_std[six_hours], X_ensemble_mean[six_hours]+X_ensemble_std[six_hours])
                    ax4.fill_between(lon, X_ensemble_mean[twelve_hours]-X_ensemble_std[twelve_hours], X_ensemble_mean[twelve_hours]+X_ensemble_std[twelve_hours])
                    ax5.fill_between(lon, X_ensemble_mean[one_day]-X_ensemble_std[one_day], X_ensemble_mean[one_day]+X_ensemble_std[one_day])
                    ax6.fill_between(lon, X_ensemble_mean[one_and_a_half_days]-X_ensemble_std[one_and_a_half_days], X_ensemble_mean[one_and_a_half_days]+X_ensemble_std[one_and_a_half_days])
                    ax7.fill_between(lon, X_ensemble_mean[half_sim]-X_ensemble_std[half_sim], X_ensemble_mean[half_sim]+X_ensemble_std[half_sim])
                    ax8.fill_between(lon, X_ensemble_mean[final_time_stamp]-X_ensemble_std[final_time_stamp], X_ensemble_mean[final_time_stamp]+X_ensemble_std[final_time_stamp])
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6),
          fancybox=True, shadow=True, ncol=legend_columns)
                    
            ############ Median lat plot ##################
            if median == True:
                X_ensemble_median = np.percentile(X_scenario, 50, axis=0)
                X_ensemble_25th = np.percentile(X_scenario, 25, axis=0)
                X_ensemble_75th = np.percentile(X_scenario, 75, axis=0)
                X_ensemble_5th = np.percentile(X_scenario, 5, axis=0)
                X_ensemble_95th = np.percentile(X_scenario, 95, axis=0)
                #if ensemoff==False: ax1.plot(lon, X_ensemble_median[0],colours[j], label='ensemble median', linewidth=5)
                ax1.plot(lon, X_ensemble_median[0],colours[j], linewidth=5)
                #ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), frameon=False, fontsize=20)
                ax2.plot(lon, X_ensemble_median[one_hour],colours[j], linewidth=5)
                ax3.plot(lon, X_ensemble_median[six_hours],colours[j], linewidth=5)
                ax4.plot(lon, X_ensemble_median[twelve_hours],colours[j], linewidth=5)
                ax5.plot(lon, X_ensemble_median[one_day],colours[j], linewidth=5)
                ax6.plot(lon, X_ensemble_median[one_and_a_half_days],colours[j], linewidth=5)
                ax7.plot(lon, X_ensemble_median[half_sim],colours[j], linewidth=5)
                ax8.plot(lon, X_ensemble_median[final_time_stamp],colours[j],label=label_lead+str(j)+", ens. median", linewidth=5)

                if ensemoff==True:
                    #ax1.text(1, 1, 'ensemble median + 25/75th + 5/95th percentiles', horizontalalignment='right',verticalalignment='bottom',transform=ax1.transAxes, fontsize=20)
                    ax1.fill_between(lon, X_ensemble_5th[0], X_ensemble_95th[0], color=colour_median, alpha=0.25)
                    ax1.fill_between(lon, X_ensemble_25th[0], X_ensemble_75th[0], color=colour_median, alpha=0.5)
                    ax2.fill_between(lon, X_ensemble_5th[one_hour], X_ensemble_95th[one_hour], color=colour_median, alpha=0.25)
                    ax2.fill_between(lon, X_ensemble_25th[one_hour], X_ensemble_75th[one_hour], color=colour_median, alpha=0.5)
                    ax3.fill_between(lon, X_ensemble_5th[six_hours], X_ensemble_95th[six_hours], color=colour_median, alpha=0.25)
                    ax3.fill_between(lon, X_ensemble_25th[six_hours], X_ensemble_75th[six_hours], color=colour_median, alpha=0.5)
                    ax4.fill_between(lon, X_ensemble_5th[twelve_hours], X_ensemble_95th[twelve_hours], color=colour_median, alpha=0.25)
                    ax4.fill_between(lon, X_ensemble_25th[twelve_hours], X_ensemble_75th[twelve_hours], color=colour_median, alpha=0.5)
                    ax5.fill_between(lon, X_ensemble_5th[one_day], X_ensemble_95th[one_day], color=colour_median, alpha=0.25)
                    ax5.fill_between(lon, X_ensemble_25th[one_day], X_ensemble_75th[one_day], color=colour_median, alpha=0.5)
                    ax6.fill_between(lon, X_ensemble_5th[one_and_a_half_days], X_ensemble_95th[one_and_a_half_days], color=colour_median, alpha=0.25)
                    ax6.fill_between(lon, X_ensemble_25th[one_and_a_half_days], X_ensemble_75th[one_and_a_half_days], color=colour_median, alpha=0.5)
                    ax7.fill_between(lon, X_ensemble_5th[half_sim], X_ensemble_95th[half_sim], color=colour_median, alpha=0.25)
                    ax7.fill_between(lon, X_ensemble_25th[half_sim], X_ensemble_75th[half_sim], color=colour_median, alpha=0.5)
                    ax8.fill_between(lon, X_ensemble_5th[final_time_stamp], X_ensemble_95th[final_time_stamp], color=colour_median, alpha=0.25)
                    ax8.fill_between(lon, X_ensemble_25th[final_time_stamp], X_ensemble_75th[final_time_stamp], color=colour_median, alpha=0.5)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6),
          fancybox=True, shadow=True, ncol=legend_columns)

        ax1.text(0.01, .8, 't=0 hours', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
        ax2.text(0.01, .8, 't=1 hour', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
        ax3.text(0.01, .8, 't=6 hours', horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes, fontsize=20)
        ax4.text(0.01, .8, 't=12 hours', horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, fontsize=20)
        ax5.text(0.01, .8, 't=1 day', horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes, fontsize=20)
        ax6.text(0.01, .8, 't=1.5 days', horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes, fontsize=20)
        ax7.text(0.01, .8, 't='+str(number_of_days//2)+' days', horizontalalignment='left', verticalalignment='center', transform=ax7.transAxes, fontsize=20)
        ax8.text(0.01, .8, 't='+str(number_of_days)+' days', horizontalalignment='left', verticalalignment='center', transform=ax8.transAxes, fontsize=20)

    ######################## 2D latplot ######################################
    
    if lat_plot == True and twoD_latplot == True and scenario_size == 0 and ensem_size == 0:
        lon=np.arange(180,-180,-360/K)
        
        plt.figure(figsize=(20,12))
        xmin = min(lon); xmax = max(lon); ymin = min(t); ymax = max(t)
        plt.imshow(X, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=colour_latplot, alpha=0.5,aspect='auto', vmin=-np.percentile(X, .9), vmax=np.percentile(X, .9))
        plt.ylabel('time (days)', fontsize=20); plt.rc('axes', labelsize=16)
        plt.xlabel('longitude', fontsize=20); plt.rc('xtick', labelsize=16)
        plt.colorbar().set_label(label='$X_k$',size=26,rotation=270,labelpad=15)
        plt.title('Lorenz 96 Model, longitudinal profile of $X_k$, F='+str(F)+", K="+str(K),size=28)
        plt.gca().invert_yaxis()
        plt.show()
        
        
    if (mean == True and ensem_size > 0):
        print("ensemble mean + standard deviation")
    if (median == True and ensem_size > 0):
        print("ensemble median + 25th/75th + 5th/95th percentiles")
    
    if is_atmos_and_ocean == True:
        print("Second Component")
        plot_Lorenz96(time_series, lat_plot, X_ocean, t, F, K, number_of_days, global_avg, extra_ts_plots, mean, median, ensemoff, twoD_latplot, label_lead = 'Second ', base_colour = False)



