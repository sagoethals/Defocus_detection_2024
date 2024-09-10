"""

A class to build the checkerboard stimulus.

Args:
    - nb_checks: the number of checks
    - binary_source_path: path to the binary source file
    - rig_nb: the set-up used for the experiments

"""

import os
from tqdm import tqdm
import numpy as np
import pickle

__all__ = ['save_obj', 'load_obj', 'load_py2_obj', 'getfiles']

class Checkerboard:

    def __init__(self, nb_checks, binary_source_path, rig_nb, repetitions, triggers):

        assert os.path.isfile(binary_source_path)

        self._nb_checks = nb_checks
        self._binary_source_path = binary_source_path
        self._rig_nb = rig_nb
        self._repetitions = repetitions
        self._triggers = triggers

        self._binary_source_file = open(self._binary_source_path, mode='rb')


    def __exit__(self, exc_type, exc_value, traceback):

        self._input_file.close()

        return

    def get_limits(self):

        return self._triggers.get_limits()

    def get_repetition_limits(self):

        start_trigger_nbs = self._repetitions.get_start_trigger_nbs(condition_nb=0)
        end_trigger_nbs = self._repetitions.get_end_trigger_nbs(condition_nb=0)

        start_sample_nbs = self._triggers.get_sample_nbs(start_trigger_nbs)
        end_sample_nbs = self._triggers.get_sample_nbs(end_trigger_nbs)

        repetition_limits = [
            (start_sample_nb, end_sample_nb)
            for start_sample_nb, end_sample_nb in zip(start_sample_nbs, end_sample_nbs)
        ]

        return repetition_limits

    def get_image_nbs(self, sample_nbs):

        trigger_nbs = self._triggers.get_trigger_nbs(sample_nbs)

        sequence_length = 300  # frames

        image_nbs = np.copy(trigger_nbs)
        for k, trigger_nb in enumerate(trigger_nbs):
            sequence_nb = trigger_nb // sequence_length
            is_in_frozen_sequence = (sequence_nb % 2) == 1
            if is_in_frozen_sequence:
                offset = 0
            else:
                offset = (sequence_nb // 2) * sequence_length
            image_nb = offset + trigger_nb % sequence_length
            image_nbs[k] = image_nb

        return image_nbs

    def _get_bit(self, bit_nb):

        byte_nb = bit_nb // 8
        self._binary_source_file.seek(byte_nb)
        byte = self._binary_source_file.read(1)
        byte = int.from_bytes(byte, byteorder='big')
        bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)

        return bit

    def get_image_shape(self):

        shape = (self._nb_checks, self._nb_checks)

        return shape

    def get_image(self, image_nb):

        shape = self.get_image_shape()
        image = np.zeros(shape, dtype=np.float)

        for i in range(0, self._nb_checks):
            for j in range(0, self._nb_checks):
                bit_nb = (self._nb_checks * self._nb_checks * image_nb) + (self._nb_checks * i) + j
                bit = self._get_bit(bit_nb)
                # Here modifications were made on 5.01.2021 by TBT to get the same polarity as the one seen on camera
                if bit == 0:
                    image[i, j] = 0.0
                elif bit == 1:
                    image[i, j] = 1.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)

        # Here modifications were made on 5.01.2021 by TBT to get the same orientation as the one seen on camera
        #image = np.flipud(image)
        #image = np.fliplr(image)
        if self._rig_nb == 2:
            image = np.rot90(image)
            image = np.flipud(image)
            
        elif self._rig_nb == 3:
            image = np.fliplr(image)

        return image

    def get_clip_shape(self, nb_images):

        shape = (nb_images,) + self.get_image_shape()

        return shape

    def get_clip(self, reference_image_nb, nb_images):

        shape = self.get_clip_shape(nb_images)
        clip = np.zeros(shape, dtype=np.float)

        for k in range(0, nb_images):
            image_nb = reference_image_nb + (k - (nb_images - 1))
            clip[k] = self.get_image(image_nb)

        return clip