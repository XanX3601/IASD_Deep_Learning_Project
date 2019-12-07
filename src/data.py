import enum

import golois
import numpy as np
import tensorflow.keras.utils as kutils
import multiprocessing

data_directory = 'data'
currently_loaded = None


class Data(enum.Enum):
    Model1466000 = (2694872, 'data/1466000.json.data')
    Model1467000 = (2790182, 'data/1467000.json.data')
    Model1468000 = (2682435, 'data/1468000.json.data')
    Model1469000 = (2791169, 'data/1469000.json.data')
    Model1470000 = (2710863, 'data/1470000.json.data')
    Model1471000 = (2703479, 'data/1471000.json.data')
    Model1472000 = (2738511, 'data/1472000.json.data')
    Model1473000 = (2520039, 'data/1473000.json.data')
    Model1474000 = (2765088, 'data/1474000.json.data')
    Model1475000 = (2790708, 'data/1475000.json.data')
    Model1476000 = (2724745, 'data/1476000.json.data')
    Model1477000 = (2680052, 'data/1477000.json.data')
    Model1478000 = (2631042, 'data/1478000.json.data')
    Model1479000 = (2641258, 'data/1479000.json.data')
    Model1480000 = (2751156, 'data/1480000.json.data')
    Model1481000 = (2808491, 'data/1481000.json.data')
    Model1482000 = (2598915, 'data/1482000.json.data')
    Model1483000 = (2610842, 'data/1483000.json.data')
    Model1484000 = (2737791, 'data/1484000.json.data')
    Model1485000 = (2745086, 'data/1485000.json.data')
    Model1486000 = (2681531, 'data/1486000.json.data')
    Model1487000 = (2714194, 'data/1487000.json.data')
    Model1488000 = (2757850, 'data/1488000.json.data')
    Model1489000 = (2742292, 'data/1489000.json.data')
    Model1490000 = (2593613, 'data/1490000.json.data')
    Model1491000 = (2879359, 'data/1491000.json.data')
    Model1492000 = (2653326, 'data/1492000.json.data')
    Model1493000 = (2695618, 'data/1493000.json.data')
    Model1494000 = (2634604, 'data/1494000.json.data')
    Model1495000 = (2668984, 'data/1495000.json.data')
    Model1496000 = (2642651, 'data/1496000.json.data')
    Model1497000 = (2860348, 'data/1497000.json.data')
    Model1498000 = (2793475, 'data/1498000.json.data')
    Model1499000 = (2772000, 'data/1499000.json.data')

    def __init__(self, nb_move, path):
        self.path = path
        self.nb_move = nb_move

    def get_random_batch(self, batch_size=128, verbose=False):
        """
        Load a random batch from self

        Args:
            bacth_size: the size of the batch to get
            verbose: if True, output details on the execution
        """
        global currently_loaded
        if currently_loaded != self:
            golois.load(self.path, verbose)
            currently_loaded = self

        input = np.empty((batch_size, 19, 19, 8), dtype=np.float32)
        policy_output = np.empty((batch_size, 361), dtype=np.float32)
        value_output = np.empty((batch_size,), dtype=np.float32)
        end = np.empty((batch_size, 19, 19, 2), dtype=np.float32)

        golois.get_random_batch(input, policy_output, value_output, end)

        return input, policy_output, value_output

    def get_batch(self, start_index, batch_size=128, verbose=False):
        """
        Load a batch from self starting at given index

        Args:
            start_index: index of the first element to get
            batch_size: the size of the bacth to get
            verbose: if True, output details on the execution
        """
        global currently_loaded
        if currently_loaded != self:
            golois.load(self.path, verbose)
            currently_loaded = self

        input = np.empty((batch_size, 19, 19, 8), dtype=np.float32)
        policy_output = np.empty((batch_size, 361), dtype=np.float32)
        value_output = np.empty((batch_size,), dtype=np.float32)
        end = np.empty((batch_size, 19, 19, 2), dtype=np.float32)

        golois.get_batch(input, policy_output, value_output, end, start_index)
        return input, policy_output, value_output

class DataSequence(kutils.Sequence):
    def __init__(self, data, batch_size, use_random_batch=False, verbose=False):
        self.data = data
        self.batch_size = batch_size
        self.lock = multiprocessing.Lock()
        self.use_random_batch = use_random_batch
        self.verbose = verbose

    def __len__(self):
        with self.lock:
            return self.data.nb_move // self.batch_size + 1

    def __getitem__(self, index):
        with self.lock:
            if self.use_random_batch:
                input, policy, value = self.data.get_random_batch(self.batch_size, self.verbose)
            else:
                input, policy, value = self.data.get_batch(index, self.batch_size, self.verbose)

            return input, {'policy': policy, 'value': value}
