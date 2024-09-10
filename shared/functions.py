#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:05:29 2022

@author: sarah
"""

import os as os
import sys
# sys.path.append('./extraction_functions') #this is where we put all the functions.py
# sys.path.append('./../../functions_PVA')  #this is where we put all the functions.py

import matplotlib.pyplot as plt
from numpy import arange
from tqdm.auto import tqdm

from shared.extract_triggers import *
from shared.save_load import *

import numpy as np
import csv
import h5py
import matplotlib.gridspec as gds
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.optimize
from scipy.cluster.hierarchy import dendrogram
from math import pi
import matplotlib.patches as pcs
import pandas as pd

import itertools
from matplotlib.pyplot import *
from numpy import *

import pickle

# Functions

__all__ = ['compute_interspike_intervals', 'compute_number_of_rpv_spikes', 'compute_refractory_period_violation', \
            'extract_trigger_trace', 'detect_onsets', 'run_minimal_sanity_check', 'clip_list', 'get_recording_offset', 
            'restrict_array', 'load_spike_times_from_phy', 'load_spike_times', 'correct_triggers_offset',
            'load_image_as_on_MEA', 'corrcoef', 'convert_pvalue_to_stars', 'load_obj']

def compute_interspike_intervals(spike_times, sampling_rate=20e+3):

    interspike_intervals = np.diff(spike_times)
    interspike_intervals = interspike_intervals.astype(np.float)
    interspike_intervals = interspike_intervals / sampling_rate

    return interspike_intervals

def compute_number_of_rpv_spikes(spike_times, duration=2.0, sampling_rate=20e+3):
    
    isis = compute_interspike_intervals(spike_times, sampling_rate=sampling_rate)
    nb_isis = np.count_nonzero(isis <= 1e-3 * duration)
    
    return nb_isis

def compute_refractory_period_violation(spike_times, duration=2.0, sampling_rate=20e+3):
    """
    spike_times : the spike times of the neuron to study
    duration : the duraiton of the refractory period, in ms
    sampling_rate : the sampling rate of the recording device
    """

    isis = compute_interspike_intervals(spike_times, sampling_rate=sampling_rate)
    nb_isis = len(isis)
    rpv = float(np.count_nonzero(isis <= 1e-3 * duration)) / float(nb_isis) *100

    return rpv

def extract_trigger_trace(input_path=None, dtype='uint16', nb_channels=256, channel_id=126):

    if not os.path.isfile(input_path):
        message = "'{}' file does not exist.".format(input_path)
        raise printer.warning(message)

    voltage_resolution = 0.1042  # µV / DC level

    # Load data.
    m = np.memmap(input_path, dtype=dtype)
    if m.size % nb_channels != 0:
        message = "number of channels is inconsistent with the data size."
        raise printer.warning(message)
    # data = m[channel_id::nb_channels]
    nb_samples = m.size // nb_channels
    data = np.empty((nb_samples,), dtype=dtype)
    for k in tqdm(range(0, nb_samples)):
        data[k] = m[nb_channels * k + channel_id]
    data = data.astype(np.float)
    data = data - np.iinfo('uint16').min + np.iinfo('int16').min
    data = data / voltage_resolution
    
    return data

def detect_onsets(data, threshold=50e+3):

    test_1 = data[:-1] < threshold
    test_2 = data[1:] >= threshold
    test = np.logical_and(test_1, test_2)

    indices = np.where(test)[0]

    test = data[indices - 1] < data[indices]
    while np.any(test):
        indices[test] = indices[test] - 1
        test = data[indices - 1] < data[indices]

    return indices

def run_minimal_sanity_check(triggers, sampling_rate=20000, maximal_jitter=0.25e-3):
    """

    :param sampling_rate: sampling rate used to acquire the data
    :param maximal_jitter: maximal jitter (in seconds) used to assert if triggers are evenly spaced
    :return:
    """

    # Check trigger statistics.
    inter_triggers = np.diff(triggers)
    inter_trigger_values, inter_trigger_counts = np.unique(inter_triggers, return_counts=True)

    index = np.argmax(inter_trigger_counts)
    inter_trigger_value = inter_trigger_values[index]

    if not np.all(np.abs(inter_trigger_values - inter_trigger_value) <= maximal_jitter * sampling_rate):
        message = "Triggers are not evenly spaced (some missing?)."
        raise UserWarning(message)
    else:
        print("Minimal sanity check ok.")

    return

def clip_list(input_list, min_value, max_value):
    
    clipped_list = input_list[input_list <= max_value]
    clipped_list = clipped_list[clipped_list >= min_value]
    
    return clipped_list

def get_recording_offset(preceding_recordings_paths, nb_electrodes=256, nb_bytes_by_datapoint=2):
    """
    :param preceding_recordings_paths. list. Contains the paths to the .raw recording files
    :return offset. int. The offset in starting time of the recording, in data points.
    To get the offset in seconds, the result should be divided by the sampling rate.
    """
    offset = 0
    for path in preceding_recordings_paths:
        file_stats = os.stat(path)
        recording_length = (file_stats.st_size)/(nb_bytes_by_datapoint*nb_electrodes)
        offset += recording_length
    return offset

def restrict_array(array, value_min, value_max):
    array = array[array>=value_min]
    array = array[array<=value_max]
    return array.tolist()

def load_spike_times_from_phy(directory, cluster_nb):
    
    all_spike_clusters = np.load(os.path.join(directory, cluster_file_name))
    all_spike_times = np.load(os.path.join(directory, spike_file_name))
    
    spike_indices = [i for i in range(len(all_spike_clusters)) if all_spike_clusters[i]==cluster_nb]
    spike_times = all_spike_times[spike_indices]
    return np.array(spike_times)

def load_spike_times(directory, file_name, cell_nb):
    path = os.path.join(directory, file_name)
    sorting_result = load_sorting_result(path)
    all_spike_times = sorting_result['spike_times']
    spike_times = all_spike_times[cell_nb]
    
    return spike_times

def correct_triggers_offset(triggers, paths=None, offset=None):
    if offset is None:
        assert paths is not None, "You must specify either the offset or the paths of the preceding recordings."
        offset = get_recording_offset(paths)            
    corrected_triggers = triggers + offset
    return corrected_triggers

def align_triggers_spikes(triggers, spike_times, sampling_rate=20000):
    # Clip the spike times to the recording time
    trigger_min = np.min(triggers)
    trigger_max = np.max(triggers)
    clipped_spike_times = clip_list(spike_times, trigger_min, trigger_max)

    # Set trigger start times to zero
    triggers_first_time = np.min(triggers)
    triggers = triggers - triggers_first_time

    # Do the same operation on spike times
    new_spike_times = clipped_spike_times - triggers_first_time
    
    # Get the values in seconds
    new_triggers = triggers/sampling_rate
    new_spike_times = new_spike_times/sampling_rate
    
    return new_triggers, new_spike_times

def extract_all_spike_times_from_phy(directory, experiment_name):
    
    all_spike_clusters = np.load(os.path.join(directory, "{}_spike_clusters.npy".format(experiment_name)))
    all_spike_times = np.load(os.path.join(directory, "{}_spike_times.npy".format(experiment_name)))
    
    spike_times = {}
    for i in tqdm(range(len(all_spike_times))):
        if all_spike_clusters[i] not in spike_times.keys():
            spike_times[all_spike_clusters[i]] = []
        spike_times[all_spike_clusters[i]] += [all_spike_times[i]]
        
    return spike_times

def evaluate_polarity(sta):
    """Separate space and time components."""

    time_width, space_height, space_width = sta.shape[0], sta.shape[1], sta.shape[2]
    rf_shape = (time_width, space_height * space_width)
    rf = np.reshape(sta, rf_shape)

    # # Remove the median.
    # rf_median = np.median(rf)
    # rf = rf - rf_median

    u, s, vh = np.linalg.svd(rf, full_matrices=False)

#     time_rf_shape = (time_width,)
#     time_rf = np.reshape(u[:, 1], time_rf_shape)  # TODO why 1 instead of 0?
#     space_rf_shape = (space_height, space_width)
#     space_rf = np.reshape(vh[1, :], space_rf_shape)  # TODO understand why 1 instead of 0?

    # Determine the cell polarity
    if np.abs(np.max(rf) - np.median(rf)) >= np.abs(np.min(rf) - np.median(rf)):
        rf_polarity = 'ON'
    else:
        rf_polarity = 'OFF'
        
    return rf_polarity

def separate_components(sta):
    """Separate space and time components."""

    time_width, space_height, space_width = sta.shape[0], sta.shape[1], sta.shape[2]
    rf_shape = (time_width, space_height * space_width)
    rf = np.reshape(sta, rf_shape)

    # # Remove the median.
    # rf_median = np.median(rf)
    # rf = rf - rf_median

    u, s, vh = np.linalg.svd(rf, full_matrices=False)
    
#     print (s)
    
    time_rf_shape = (time_width,)
    time_rf = np.reshape(u[:, 1], time_rf_shape)  # TODO why 1 instead of 0?
    space_rf_shape = (space_height, space_width)
    space_rf = np.reshape(vh[1, :], space_rf_shape)  # TODO understand why 1 instead of 0?

    # Determine the cell polarity
    if np.abs(np.max(rf) - np.median(rf)) >= np.abs(np.min(rf) - np.median(rf)):
        rf_polarity = 'ON'
    else:
        rf_polarity = 'OFF'
        
    # Determine the spatial RF polarity
    if np.abs(np.max(space_rf) - np.median(space_rf) >= np.abs(np.min(space_rf) - np.median(space_rf))):
        space_rf_polarity = 'ON'
    else:
        space_rf_polarity = 'OFF'
        
    # Determine the temporal RF polarity
    if np.abs(np.max(time_rf) - np.median(time_rf) >= np.abs(np.min(time_rf) - np.median(time_rf))):
        time_rf_polarity = 'ON'
    else:
        time_rf_polarity = 'OFF'
        
    # Reverse components (if necessary). WHY ???
    if rf_polarity != space_rf_polarity:
        space_rf = - space_rf
        
    if rf_polarity != time_rf_polarity:
        time_rf = - time_rf

    return time_rf, space_rf

def fit_gaussian(space_sta):
    """Fit a 2D Gaussian."""

    if np.abs(np.max(space_sta) - np.median(space_sta)) >= np.abs(np.min(space_sta) - np.median(space_sta)):
        polarity = 'ON'
    else:
        polarity = 'OFF'
        
    # TODO add smoothing and thresholding steps

    # Gaussian fit.
    def gaussian(p, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = p
        xo = float(xo)
        yo = float(yo)
        a = + (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = - (np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = + (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
        return g.ravel()

    ## Create x and y indices
    x = np.linspace(0, space_sta.shape[1] - 1, space_sta.shape[1])  # TODO define in the constructor.
    y = np.linspace(0, space_sta.shape[0] - 1, space_sta.shape[0])  # TODO define in the constructor.
    x, y = np.meshgrid(x, y)
    ## Do fitting.
    if polarity == 'ON':
        a0 = +3.0
        yo, xo = np.unravel_index(space_sta.argmax(), space_sta.shape)
    elif polarity == 'OFF':
        a0 = -3.0
        yo, xo = np.unravel_index(space_sta.argmin(), space_sta.shape)
    else:
        raise NotImplementedError
    sigma_x0, sigma_y0 = 5.0, 5.0
    theta0 = 0.0
    offset0 = 0.0
    initial_guess = (a0, xo, yo, sigma_x0, sigma_y0, theta0, offset0)
    try:
        popt, pcov = scipy.optimize.curve_fit(gaussian, (x, y), space_sta.ravel(), p0=initial_guess)
    except RuntimeError:
        popt = [0,0,0,0,0,0,0]
        pcov = [0,0,0,0,0,0,0]
    ## Fitted data.
    # fitted_space_sta = gaussian((x, y), *initial_guess)
    fitted_space_sta = gaussian((x, y), *popt)
    fitted_space_sta = np.reshape(fitted_space_sta, space_sta.shape)

    return x, y, fitted_space_sta, popt

def get_ellipse_parameters(parameters,factor=1.5,printing=False):
    """Get ellipse parameters."""

    amplitude, x0, y0, sigma_x, sigma_y, theta, offset = parameters
    if printing:
        print("sigma_x",sigma_x, 'pixel,')
        print("sigma_y",sigma_y, 'pixel,')
        print ("amplitude:", amplitude)
    width = factor * 2.0 * sigma_x
    height = factor * 2.0 * sigma_y
    angle = - np.rad2deg(theta)

    return x0, y0, width, height, angle

def get_ellipse(params, factor=1.5):

    x0, y0, width, height, angle, linestyle = get_ellipse_parameters(params,factor=factor)
    ellipse = lambda color=None: pcs.Ellipse((x0, y0), width, height, angle=angle, linestyle=linestyle, color='k', fill=False)

    return ellipse

def plot_ellipse(ax, parameters):
    from math import pi, cos, sin

    u = parameters[0]       #x-position of the center
    v = parameters[1]       #y-position of the center
    a = parameters[2]        #radius on the x-axis
    b = parameters[3]       #radius on the y-axis
    t_rot = parameters[4]  #rotation angle

    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
         #u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
         #2-D rotation matrix

    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

    return ax.plot((u+Ell_rot[0,:], v+Ell_rot[1,:]))

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def get_cell_rpvs(cell_lists,rpv_len,spike_times,spike_clusters):

    PLOT=False
    
    for good_clusters in cell_lists:
        cell_data ={}
        # refractory time duration lenght
        

        min_rpv=100
        max_rpv=0
        cell_count=0
        avg_rpv=0
        rpv_list=[]
        for cell_nb in tqdm(good_clusters[:]):
        #     cell_nb=373
            cell_data[cell_nb] = {}
            #spike_times = all_spike_times[cell_nb]
            sp_times = spike_times[spike_clusters==int(cell_nb)]#/fs
            interspike_intervals = compute_interspike_intervals(sp_times)
            rpv = compute_refractory_period_violation(sp_times,duration=rpv_len)
            nb_rpv = compute_number_of_rpv_spikes(sp_times,duration=rpv_len)

            if min_rpv>rpv:min_rpv=rpv
            if max_rpv<rpv:max_rpv=rpv

            #if max_rpv>0.21: print(cell_nb)

            cell_data[cell_nb]["nb_spikes"] = len(spike_times)
            cell_data[cell_nb]["isi"] = interspike_intervals
            cell_data[cell_nb]["rpv"] = rpv
            cell_data[cell_nb]["nb_rpv_spikes"] = nb_rpv

            #print('Cell ',cell_nb, '  Violations % ',np.round(rpv,2), '  N= ',nb_rpv)#nb_rpv,len(sp_times),rpv
            #------------------------------------------
            if PLOT:
                fig= figure(figsize=(3,6))
                hist(interspike_intervals,np.linspace(0,0.05,50))
                xlim([0,0.05])
                axvline(rpv_len/1000,color='k') 
                title('Cell ' + str(cell_nb) + '  Violations % '+str(np.round(rpv,2)) + '  N= '+str(nb_rpv))
            #------------------------------------------
            cell_count+=1
            avg_rpv+=rpv
            rpv_list.append(rpv)

        avg_rpv=avg_rpv/cell_count
        rpv_list=np.array(rpv_list)
        
        print('Min rpv:' ,np.round(min_rpv,2),'%')
        print('Avg rpv:' ,np.round(avg_rpv,2),'%')
        print('Median rpv:' ,np.round(np.median(rpv_list),2),'%')
        print('Max rpv:' ,np.round(max_rpv,2),'%')
        
    return cell_data

def load_image_as_on_MEA(path, flip_and_rotate=True):
    '''
    A function that applies the same transforme as the camera on the set-up. 
    It sets the image in the right orientation.
    '''
    if flip_and_rotate:
        img = plt.imread(path)
        img = np.rot90(img)
        img = np.flipud(img)
    else:
        img = plt.imread(path)
    return img

def plot_scale_bar(
        axis, scale=None, unit=None, loc='lower right', orientation='horizontal', with_label=True, color='black'):

    pad = 0.06

    # fontsize = 'normal'
    # text_offset = 8  # pt
    fontsize = 'small'
    text_offset = 6  # pt

    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()

    # Find scale automatically (if necessary).
    if scale is None:
        width = (1.0 - 2.0 * pad) * (x_max - x_min)
        height = (1.0 - 2.0 * pad) * (y_max - y_min)
        if orientation == 'horizontal':
            size = width
        elif orientation == 'vertical':
            size = height
        else:
            raise ValueError("unexpected value: {}".format(orientation))
        order = int(np.floor(np.log10(size)))
        scale = 10 ** order
        factor = 1
    else:
        order = int(np.floor(np.log10(scale)))
        factor = scale / 10 ** order

    map_order = {
        -9: (1, 'n'),
        -8: (10, 'n'),
        -7: (100, 'n'),
        -6: (1, 'µ'),
        -5: (10, 'µ'),
        -4: (100, 'µ'),
        -3: (1, 'm'),
        -2: (10, 'm'),
        -1: (100, 'm'),
        0: (1, ''),
        1: (10, ''),
        2: (100, ''),
        3: (1, 'k'),
        4: (10, 'k'),
        5: (100, 'k'),
        6: (1, 'M'),
        7: (10, 'M'),
        8: (100, 'M'),
    }
    if unit is None:
        text = "{}".format(scale)
    else:
        factor_, prefix = map_order[order]
        factor = factor * factor_
        factor = int(factor) if factor.is_integer() else factor
        text = "{} {}{}".format(factor, prefix, unit)

    plot_kwargs = {
        'color': color,
        # 'linewidth': 3.5,
        'linewidth': 2.0,
        'solid_capstyle': 'butt',  # instead of 'projection (default)'
    }

    if loc == 'lower right':
        if orientation == 'horizontal':
            x_ref = x_max - pad * (x_max - x_min) - 0.5 * scale
            y_ref = y_min + pad * (y_max - y_min)
            axis.plot([x_ref - 0.5 * scale, x_ref + 0.5 * scale], [y_ref, y_ref], **plot_kwargs)
            if with_label:
                axis.annotate(
                    text, (x_ref, y_ref), xytext=(0, +text_offset),
                    xycoords='data', textcoords='offset points',
                    color=color, horizontalalignment='center', verticalalignment='center', rotation='horizontal',
                    fontsize=fontsize
                )
        elif orientation == 'vertical':
            x_ref = x_max - pad * (x_max - x_min)
            y_ref = y_min + pad * (y_max - y_min) + 0.5 * scale
            axis.plot([x_ref, x_ref], [y_ref - 0.5 * scale, y_ref + 0.5 * scale], **plot_kwargs)
            if with_label:
                axis.annotate(
                    text, (x_ref, y_ref), xytext=(-text_offset, 0),
                    xycoords='data', textcoords='offset points',
                    color=color, horizontalalignment='center', verticalalignment='center', rotation='vertical',
                    fontsize=fontsize
                )
        else:
            raise ValueError("unexpected value: {}".format(orientation))
    elif loc == 'lower left':
        if orientation == 'horizontal':
            # TODO remove the following lines?
            # x_ref = x_min + pad * (x_max - x_min) + 0.5 * scale
            # y_ref = y_min + pad * (y_max - y_min)
            # axis.plot([x_ref - 0.5 * scale, x_ref + 0.5 * scale], [y_ref, y_ref], **plot_kwargs)
            # if with_label:
            #     axis.annotate(
            #         text, (x_ref, y_ref), xytext=(0, 8),
            #         xycoords='data', textcoords='offset points',
            #         horizontalalignment='center', verticalalignment='center', rotation='horizontal',
            #     )
            x_ref = x_min + pad * (x_max - x_min)
            y_ref = y_min + pad * (y_max - y_min)
            axis.plot([x_ref + 0.0 * scale, x_ref + 1.0 * scale], [y_ref, y_ref], **plot_kwargs)
            if with_label:
                axis.annotate(
                    text, (x_ref, y_ref), xytext=(0, +text_offset),
                    xycoords='data', textcoords='offset points',
                    color=color, horizontalalignment='left', verticalalignment='center', rotation='horizontal',
                    fontsize=fontsize
                )
        elif orientation == 'vertical':
            x_ref = x_min + pad * (x_max - x_min)
            y_ref = y_min + pad * (y_max - y_min) + 0.5 * scale
            axis.plot([x_ref, x_ref], [y_ref - 0.5 * scale, y_ref + 0.5 * scale], **plot_kwargs)
            if with_label:
                axis.annotate(
                    text, (x_ref, y_ref), xytext=(+text_offset, 0),
                    xycoords='data', textcoords='offset points',
                    color=color, horizontalalignment='center', verticalalignment='center', rotation='vertical',
                    fontsize=fontsize
                )
        else:
            raise ValueError("unexpected value: {}".format(orientation))
    else:
        raise NotImplementedError()

    return

def corrcoef(x, y):
    """Return Pearson product-moment correlations coefficients.

    This is a wrapper around `np.corrcoef` to avoid:
        `RuntimeWarning: invalid value encountered in true_divide`.
    """

    assert len(x) > 0, len(x)
    assert len(y) > 0, len(y)
    assert len(x) == len(y), (len(x), len(y))

    is_x_deterministic = np.all(x == x[0])  # i.e. array filled with a unique value
    is_y_deterministic = np.all(y == y[0])  # i.e. array filled with a unique value
    if is_x_deterministic and is_y_deterministic:
        r = 1.0
    elif is_x_deterministic or is_y_deterministic:
        r = 0.0
    else:
        r = np.corrcoef(x, y)[0, 1]

    return r

def convert_pvalue_to_stars(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"



def load_obj(name ):
    if name[-4:]=='.pkl':
        name = name[:-4]
    #~ try:
        #~ return pk5.dumps(name+'pkl', protocol=5)
    #~ except:
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)