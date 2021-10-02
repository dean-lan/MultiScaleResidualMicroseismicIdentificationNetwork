import os
import numpy as np
import pandas as pd


class DataSetLoader(object):
    def __init__(self):
        super(DataSetLoader, self).__init__()

    @staticmethod
    def _do_augment(x, y, augmentations, max_shift, padding=0.0):
        x_size = len(x)
        augmented_x = []
        for _ in range(augmentations):
            x_to_shift_index = np.random.randint(low=0, high=x_size - 1)
            x_to_shift = x[x_to_shift_index]
            num_to_shift = np.random.randint(low=-max_shift, high=max_shift)

            if num_to_shift == 0:
                augmented_x.append(x_to_shift)
                continue

            paddings = np.ones(abs(num_to_shift)) * padding
            if num_to_shift > 0:
                # move time window to the right
                shifted_x = np.concatenate((x_to_shift[num_to_shift:], paddings))
            if num_to_shift < 0:
                # move time window to the left
                shifted_x = np.concatenate((paddings, x_to_shift[:num_to_shift]))
            augmented_x.append(shifted_x)
        augmented_x = np.asarray(augmented_x)
        augmented_y = np.asarray([y[0]] * augmentations)
        return augmented_x, augmented_y

    @staticmethod
    def normalization(x, xmin=0.0, xmax=1.0):
        # x = (x - np.min(x, axis=1, keepdims=True)) / (np.max(x, axis=1, keepdims=True) -
        #                                               np.min(x, axis=1, keepdims=True) + np.finfo(x.dtype).eps)
        # x = (xmax - xmin) * x + xmin

        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        x = (x - x_mean) / x_std

        return x

    @staticmethod
    def augment(**kwargs):
        max_shift = kwargs.get('max_shift', 200)
        shuffle = kwargs.get('shuffle', True)
        data_to_augment_x = kwargs.get('data_to_augment_x')
        data_to_augment_y = kwargs.get('data_to_augment_y')

        true_index = np.nonzero(data_to_augment_y > 0)[0]
        false_index = np.nonzero(data_to_augment_y < 1)[0]

        augmentations = len(true_index) - len(false_index)

        if augmentations > 0:
            augmented_x, augmented_y = DataSetLoader._do_augment(data_to_augment_x[false_index],
                                                                 data_to_augment_y[false_index],
                                                                 augmentations, max_shift)
        if augmentations < 0:
            augmented_x, augmented_y = DataSetLoader._do_augment(data_to_augment_x[true_index],
                                                                 data_to_augment_y[true_index],
                                                                 -augmentations, max_shift)
        if augmentations == 0:
            augmented_data_x = np.copy(data_to_augment_x)
            augmented_data_y = np.copy(data_to_augment_y)
        else:
            augmented_data_x = np.vstack((data_to_augment_x, augmented_x))
            augmented_data_y = np.concatenate((data_to_augment_y, augmented_y))

        if shuffle:
            seed = np.random.randint(low=0, high=191024)
            np.random.seed(seed=seed)
            np.random.shuffle(augmented_data_x)
            np.random.seed(seed=seed)
            np.random.shuffle(augmented_data_y)

        return augmented_data_x, augmented_data_y

    @staticmethod
    def split(**kwargs):
        # Only used for wilkins datasets
        data_x = None
        data_y = []

        n_channel = kwargs.get('n_channel', 1)
        data_to_split_x = kwargs.get('data_to_split_x')
        data_to_split_y = kwargs.get('data_to_split_y')

        data_size = data_to_split_x.shape[0]

        if n_channel == 1:
            return np.copy(data_to_split_x), np.copy(data_to_split_y)

        for index in range(data_size):
            raw_x = data_to_split_x[index]
            split_x = np.reshape(raw_x, newshape=(n_channel, int(raw_x.shape[0] / n_channel)))
            if data_x is None:
                data_x = split_x
            else:
                data_x = np.vstack((data_x, split_x))

            raw_y = data_to_split_y[index]
            data_y += [raw_y] * n_channel
        split_data_x = np.asarray(data_x)
        split_data_y = np.asarray(data_y)
        return split_data_x, split_data_y

    @staticmethod
    def load_from_file(file_name):
        dset = pd.read_csv(file_name, header=None)
        raw_data = dset.to_numpy()
        raw_data_x = raw_data[:, :-1]
        raw_data_y = raw_data[:, -1]

        return raw_data_x, raw_data_y

