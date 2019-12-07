import src
import argparse
import os

import tensorflow.keras.optimizers as koptimizers
import tensorflow.keras.losses as klosses

models = {'basic': src.basic_model}

parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('model', choices=models.keys(), help='the name of the model to train')
parser.add_argument('--verbose', '-v', action='store_true', help='if set, output details on the execution')
parser.add_argument('--tf-log-level', default='3', choices=[str(i) for i in range(4)], help='tensorflow minimum cpp log level')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level

model = models[args.model]()

if args.verbose:
    model.summary()

optimizer = koptimizers.SGD(nesterov=True)
model.compile(optimizer=optimizer, loss={'value': klosses.mean_squared_error, 'policy': klosses.categorical_crossentropy}, metrics=['accuracy'])

data_sequence = src.DataSequence(src.Data.Model1498000, batch_size=2048, verbose=True)
data_sequence_valid = src.DataSequence(src.Data.Model1499000, batch_size=2048, verbose=True)

model.fit_generator(data_sequence, epochs=3, verbose=1, validation_data=data_sequence_valid, workers=8, use_multiprocessing=True, max_queue_size=128)
