import enum

import golois
import numpy as np

data_directory = 'data'
currently_loaded = None


class Data(enum.Enum):
    Model1471000 = (13973, 'data/1471000.json.data')
    Model1472000 = (14184, 'data/1472000.json.data')
    Model1496000 = (13086, 'data/1496000.json.data')
    Model1494000 = (13336, 'data/1494000.json.data')
    Model1486000 = (13417, 'data/1486000.json.data')
    Model1483000 = (13332, 'data/1483000.json.data')
    Model1480000 = (13907, 'data/1480000.json.data')
    Model1499000 = (14690, 'data/1499000.json.data')
    Model1466000 = (13185, 'data/1466000.json.data')
    Model1481000 = (14301, 'data/1481000.json.data')
    Model1469000 = (14053, 'data/1469000.json.data')
    Model1476000 = (14254, 'data/1476000.json.data')
    Model1475000 = (14260, 'data/1475000.json.data')
    Model1490000 = (13135, 'data/1490000.json.data')
    Model1493000 = (13700, 'data/1493000.json.data')
    Model1492000 = (13448, 'data/1492000.json.data')
    Model1485000 = (13808, 'data/1485000.json.data')
    Model1478000 = (13413, 'data/1478000.json.data')
    Model1468000 = (13476, 'data/1468000.json.data')
    Model1477000 = (13926, 'data/1477000.json.data')
    Model1470000 = (13834, 'data/1470000.json.data')
    Model1482000 = (13223, 'data/1482000.json.data')
    Model1497000 = (14446, 'data/1497000.json.data')
    Model1467000 = (13846, 'data/1467000.json.data')
    Model1474000 = (14091, 'data/1474000.json.data')
    Model1498000 = (14585, 'data/1498000.json.data')
    Model1479000 = (13239, 'data/1479000.json.data')
    Model1488000 = (13907, 'data/1488000.json.data')
    Model1491000 = (14444, 'data/1491000.json.data')
    Model1487000 = (13378, 'data/1487000.json.data')
    Model1484000 = (13939, 'data/1484000.json.data')
    Model1495000 = (13350, 'data/1495000.json.data')
    Model1489000 = (13912, 'data/1489000.json.data')
    Model1473000 = (12751, 'data/1473000.json.data')

    def __init__(self, nb_game, path):
        self.path = path
        self.nb_game = nb_game

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
