import os
from pathlib import Path

import numpy as np


def absolute_path(directory: str):
    """
    Gets absolute path on any os.

    Args:
        directory:

    Returns:

    """
    return os.path.join(str(Path(__file__).parents[1]), directory)


class Normalizer:
    FULL_PATH = absolute_path('normalization_weights')
    MIN_MAXES = {}

    @staticmethod
    def normalize(state: np.array, name: str) -> np.array:
        """
        Expecting the state to be:
        N x F where N is the number of entries, and F is the
        features. This method will normalize column wise.

        Creates a cache for later uses. Allows for automatically
        learning the min and maxes. It is recommended that the name include the shape of
        the state variable.

        Args:
            state: N x F where N is the number of entries, and F is the features
            name: Name of the saved file

        Returns:

        """
        assert len(state.shape) > 1, f'State with shape {state.shape} needs to be 2D'
        assert state.shape[1] >= 1, f'State with shape {state.shape} needs >=1 features'

        if not os.path.exists(Normalizer.FULL_PATH):
            os.mkdir(Normalizer.FULL_PATH)

        filepath = os.path.join(Normalizer.FULL_PATH, name)
        if name not in Normalizer.MIN_MAXES and os.path.exists(filepath):
            Normalizer.MIN_MAXES[name] = np.load(filepath)
            assert Normalizer.MIN_MAXES[name].shape[1] == state.shape[1], 'The loaded state file, ' \

        # Init the fields if needed
        if name not in Normalizer.MIN_MAXES:
            Normalizer.MIN_MAXES[name] = np.zeros(np.array(state).shape)
            Normalizer.MIN_MAXES[name] = np.vstack((Normalizer.MIN_MAXES[name], np.min(state, axis=0)))
        else:
            # If it is not none, look at the robot state, the current min max
            min_max_slice = np.vstack((state, Normalizer.MIN_MAXES[name]))
            Normalizer.MIN_MAXES[name][0] = np.max(min_max_slice, axis=0)
            Normalizer.MIN_MAXES[name][1] = np.min(min_max_slice, axis=0)

        # Normalize the states
        norm_state = np.divide(state - Normalizer.MIN_MAXES[name][1],
                 (Normalizer.MIN_MAXES[name][0] - Normalizer.MIN_MAXES[name][1]))
        norm_state[np.isnan(norm_state)] = 1

        return norm_state

    @staticmethod
    def cache_all_min_maxes():
        if not os.path.exists(Normalizer.FULL_PATH):
            os.mkdir(Normalizer.FULL_PATH)

        for key in Normalizer.MIN_MAXES:
            Normalizer.cache_min_max(key)

    @staticmethod
    def cache_min_max(name):
        if not os.path.exists(Normalizer.FULL_PATH):
            os.mkdir(Normalizer.FULL_PATH)

        filepath = os.path.join(Normalizer.FULL_PATH, name)
        np.save(filepath, Normalizer.MIN_MAXES[name])
