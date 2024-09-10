#Get triggers

import os as os
from tqdm import tqdm

import logging
#import matplotlib.pyplot as plt

import numpy as np
from colorama import Fore

__all__ = ['_detect_onsets_filtered', 'extract_filtered', 'run_minimal_sanity_check', '_detect_onsets', 'extract']

class Printer:

    def __init__(self, name, logger=None):

        self._name = name
        self._logger = logger

    @staticmethod
    def print(message):

        string = "{}{}{}".format(Fore.RESET, message, Fore.RESET)
        print(string)

    def info(self, message):

        string = "{}Info: {}{}".format(Fore.RESET, message, Fore.RESET)
        print(string)
        self._logger.info(message)

    def debug(self, message):

        string = "{}Debug: {}{}".format(Fore.BLUE, message, Fore.RESET)
        print(string)
        self._logger.debug(message)

    def warning(self, message):

        string = "{}Warning: {}{}".format(Fore.YELLOW, message, Fore.RESET)
        print(string)
        self._logger.warning(message)

        return AbortError(self._name)

    def error(self, message):

        string = "{}Error: {}{}".format(Fore.RED, message, Fore.RESET)
        print(string)
        self._logger.error(message)

        return AbortError(self._name)

class AbortError(RuntimeError):

    pass


def get_logger(name):

    logger = logging.getLogger(name)

    return logger


def get_printer(name):

    logger = get_logger(name)
    a = Printer(name, logger=logger)

    return a


logger = get_logger(__name__)
printer = get_printer(__name__)


##############################
# DETECT ONSETS
##############################
def _detect_onsets_filtered(data, threshold=50e+3):

    test_1 = data[:-1] < threshold
    test_2 = data[1:] >= threshold
    test = np.logical_and(test_1, test_2)

    indices = np.where(test)[0]

    test = data[indices - 1] < data[indices]
    while np.any(test):
        indices[test] = indices[test] - 1
        test = data[indices - 1] < data[indices]
    #------    
    test_1 = data[:-1] > -threshold
    test_2 = data[1:] <= -threshold
    test = np.logical_and(test_1, test_2)

    indices2 = np.where(test)[0]

    test = data[indices2 - 1] < data[indices2]
    while np.any(test):
        indices2[test] = indices2[test] - 1
        test = data[indices2 - 1] < data[indices2]

    return np.append(indices2[0]-1,indices[1:]+2), indices2[0]

##############################
# EXTRACT FILTERED
##############################
def extract_filtered(input_path=None, dtype='uint16', nb_channels=256, channel_id=126, output_path=None):

    if not os.path.isfile(input_path):
        message = "'{}' file does not exist.".format(input_path)
        raise printer.warning(message)

    voltage_resolution = 0.1042  # ÂµV / DC level

    # Load data.
    m = np.memmap(input_path, dtype=dtype)#, offset=1902)

    if m.size % nb_channels != 0:
        message = "number of channels is inconsistent with the data size."
        raise printer.warning(message)
    # data = m[channel_id::nb_channels]
    nb_samples = m.size // nb_channels
    print('samples: ',nb_samples, '   time: ',str(nb_samples/20000), ' s')
    data = np.empty((nb_samples,), dtype=dtype)
    
    
    ##nb_samples = 20000*400
    
    for k in tqdm(range(0, nb_samples)):
        data[k] = -m[nb_channels * k + channel_id]
    data = data.astype(np.float)
    data = data - np.iinfo('uint16').min + np.iinfo('int16').min
    data = data / voltage_resolution

    indices,indices2 = _detect_onsets_filtered(data)

    if output_path is not None:
        if os.path.isfile(output_path):
            message = "'{}' file already exists.".format(output_path)
            raise printer.warning(message)  # TODO add prompt.
        dtype = 'int32'
        shape = indices.shape
        m = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)
        indices = indices.astype(dtype)
        m[:] = indices[:]
        message = "triggers saved in '{}'".format(output_path)
        printer.info(message)

    return indices, data, nb_samples/20000

##############################
# RUN MINIMAL SANITY CHECK
##############################
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
    else:extract
    
##############################
# DETECT ONSETS
##############################
def _detect_onsets(data, threshold=50e+3):

    test_1 = data[:-1] < threshold
    test_2 = data[1:] >= threshold
    test = np.logical_and(test_1, test_2)

    indices = np.where(test)[0]

    test = data[indices - 1] < data[indices]
    while np.any(test):
        indices[test] = indices[test] - 1
        test = data[indices - 1] < data[indices]

    return indices

##############################
# EXTRACT
##############################
def extract(input_path=None, dtype='uint16', nb_channels=256, channel_id=126, output_path=None):

    if not os.path.isfile(input_path):
        message = "'{}' file does not exist.".format(input_path)
        raise printer.warning(message)

    voltage_resolution = 0.1042  # µV / DC level

    # Load data.
    m = np.memmap(input_path, dtype=dtype)#, offset=1902)
    if m.size % nb_channels != 0:
        message = "number of channels is inconsistent with the data size."
        raise printer.warning(message)
    # data = m[channel_id::nb_channels]
    nb_samples = m.size // nb_channels
    data = np.empty((nb_samples,), dtype=dtype)

    print ('Nb samples', nb_samples)
    
    for k in tqdm(range(0, nb_samples)):
        data[k] = m[nb_channels * k + channel_id]
    data = data.astype(np.float)
    data = data - np.iinfo('uint16').min + np.iinfo('int16').min
    data = data / voltage_resolution

    indices = _detect_onsets(data, threshold=150e+3)
    print ('Len indices:', len(indices))

    if output_path is not None:
        if os.path.isfile(output_path):
            message = "'{}' file already exists.".format(output_path)
            raise printer.warning(message)  # TODO add prompt.
        dtype = 'int32'
        shape = indices.shape
        m = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)
        indices = indices.astype(dtype)
        m[:] = indices[:]
        message = "triggers saved in '{}'".format(output_path)
        printer.info(message)

    return indices,data,nb_samples/20000
