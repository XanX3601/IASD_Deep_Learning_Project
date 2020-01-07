import argparse
import os
import random
import time

import tensorflow as tf
import tqdm

import src


def get_date_str():
    return time.strftime('%d/%m/%Y - %H:%M:%S')


def data_type(string):
    data_number = int(string)
    data_name = 'Model{}'.format(data_number)
    try:
        data = src.Data[data_name]
    except KeyError:
        raise argparse.ArgumentTypeError('{} does not exist'.format(data_name))
    return data


models = {
    'basic': src.basic_model,
    'resnet': src.resnet,
}

loss_value_functions = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'bce': tf.keras.losses.BinaryCrossentropy(from_logits=True),
}

backup_dir = '{}backups/'.format(src.models_dir)

parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('model', choices=models.keys(),
                    help='the name of the model to train')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='if set, output details on the execution')
parser.add_argument('--tf-log-level', default='3',
                    choices=[str(i) for i in range(4)], help='tensorflow minimum cpp log level')
parser.add_argument('--validation-data', '-vd', default=src.Data.Model1499000,
                    type=data_type, help='the data number used to make validation dataset')
parser.add_argument('--validation-dataset-size', '-vds', default=100000,
                    type=int, help='the size of the validation dataset')
parser.add_argument('--batch-size', '-bs', default=64,
                    type=int, help='mini batch size')
parser.add_argument('--epoch', '-e', default=100,
                    type=int, help='number of epochs')
parser.add_argument('--train-dataset-size', '-tds', default=100000,
                    type=int, help='the size of the train dataset')
parser.add_argument('--train-dataset-epoch', '-tde',  default=1, type=int, help='the number of epoch between train data set generation')
parser.add_argument('--train-logs', '-tl', default='{}train_logs.csv'.format(
    src.results_dir), help='path to train logs file')
parser.add_argument('--train-logs-append', '-tla', action='store_true', help='if set, happen to  train logs instead of replacing')
parser.add_argument(
    '--model-file', '-mf', default='{}model.h5'.format(src.models_dir), help='path to the model file')
parser.add_argument('--value-loss', '-vl', default='mse',
                    choices=loss_value_functions.keys(), help='the loss to use on value')
parser.add_argument('--backup', action='store_true',
                    help='if set, backup model at every epoch')
parser.add_argument('--weights-in', '-wi', help='an h5 file from which load the weights before training')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level

# get model
if args.verbose:
    print('{}: Begin "Model Creation"'.format(get_date_str()))
    start = time.time()

model = models[args.model]()

if args.weights_in is not None:
    model.load_weights(args.weights_in)

if args.verbose:
    print('{}: Done "Model Creation" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# prepare data
if args.verbose:
    print('{}: Begin "Data Preparation"'.format(get_date_str()))
    start = time.time()

datas = list(src.Data)
datas_next_start_index = {data: 0 for data in datas}
data_index = len(datas) - 1
datas_weight = [index for index in range(len(datas))]

if args.verbose:
    print('{}: Done "Data Preparation" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# create validation dataset
if args.verbose:
    print('{}: Begin "Create Validation Dataset"'.format(get_date_str()))
    start = time.time()

validation_input, validation_policy, validation_value = args.validation_data.get_batch(
    args.validation_data.size - args.validation_dataset_size,
    args.validation_dataset_size
)

validation_dataset = tf.data.Dataset.from_tensor_slices((
    validation_input,
    {src.policy_output_name: validation_policy,
        src.value_output_name: validation_value}
)).batch(args.batch_size)

if args.verbose:
    print('{}: Done "Create Validation Dataset" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# create loss functions
if args.verbose:
    print('{}: Begin "Create Loss Functions"'.format(get_date_str()))
    start = time.time()

loss_policy_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_value_fn = loss_value_functions[args.value_loss]

if args.verbose:
    print('{}: Done "Create Loss Functions" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# create optimizer
if args.verbose:
    print('{}: Begin "Create Optimizer"'.format(get_date_str()))
    start = time.time()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, nesterov=True, momentum=0.5)

if args.verbose:
    print('{}: Done "Create Optimizer" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# open train log files
if args.verbose:
    print('{}: Begin "Open Train Logs File"'.format(get_date_str()))
    start = time.time()

src.create_dir(src.results_dir)
train_logs = open(args.train_logs, 'a' if args.train_logs_append else 'w')
if not args.train_logs_append:
    train_logs.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
        src.train_logs_epoch,
        src.train_logs_start_time,
        src.train_logs_end_time,
        src.train_logs_data,
        src.train_logs_start_index,
        src.TrainLogsMetrics.train_loss,
        src.TrainLogsMetrics.train_loss_policy,
        src.TrainLogsMetrics.train_loss_value,
        src.TrainLogsMetrics.train_accuracy_policy,
        src.TrainLogsMetrics.train_accuracy_value,
        src.TrainLogsMetrics.validation_loss,
        src.TrainLogsMetrics.validation_loss_policy,
        src.TrainLogsMetrics.validation_loss_value,
        src.TrainLogsMetrics.validation_accuracy_policy,
        src.TrainLogsMetrics.validation_accuracy_value,
    ))

if args.verbose:
    print('{}: Done "Open Train Logs File" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# create backup dir
if args.backup:
    src.create_dir(backup_dir)

# keep track of the best models
if args.backup:
    best_accuracy_policy = None
    best_accuracy_value = None

# epoch loop
epoch_tqdm = tqdm.trange(args.epoch, desc='Epoch', unit='epoch', disable=True)
for epoch in epoch_tqdm:
    if args.verbose:
        epoch_tqdm.write('epoch {}'.format(epoch))

    # create metrics
    epoch_start_time = time.time()
    epoch_train_loss = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.train_loss)
    epoch_train_loss_policy = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.train_loss_policy)
    epoch_train_loss_value = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.train_loss_value)
    epoch_train_accuracy_policy = tf.keras.metrics.CategoricalAccuracy(
        name=src.TrainLogsMetrics.train_accuracy_policy)
    epoch_train_accuracy_value = tf.keras.metrics.BinaryAccuracy(
        src.TrainLogsMetrics.train_accuracy_value)
    epoch_validation_loss = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.validation_loss)
    epoch_validation_loss_policy = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.validation_loss_policy)
    epoch_validation_loss_value = tf.keras.metrics.Mean(
        name=src.TrainLogsMetrics.validation_loss_value)
    epoch_validation_accuracy_policy = tf.keras.metrics.CategoricalAccuracy(
        name=src.TrainLogsMetrics.validation_accuracy_policy)
    epoch_validation_accuracy_value = tf.keras.metrics.BinaryAccuracy(
        src.TrainLogsMetrics.train_accuracy_value)

    # prepare train dataset
    if epoch % args.train_dataset_epoch == 0:
        if args.verbose:
            epoch_tqdm.write(
                '    {}: Begin "Create Training Dataset"'.format(get_date_str()))
            start = time.time()

        try:
            del train_input, train_policy, train_value, train_dataset
        except:
            pass

        data = random.choices(datas, datas_weight)[0]

        epoch_start_index = datas_next_start_index[data]

        train_input, train_policy, train_value = data.get_batch(
            datas_next_start_index[data], args.train_dataset_size)
        train_dataset = tf.data.Dataset.from_tensor_slices((
            train_input,
            {src.policy_output_name: train_policy,
                src.value_output_name: train_value}
        )).batch(args.batch_size)

        if args.verbose:
            epoch_tqdm.write('    {}: Info Data "{}"; index {}'.format(
                get_date_str(), data.name, datas_next_start_index[data]))

        datas_next_start_index[data] += args.train_dataset_size
        if data == args.validation_data and datas_next_start_index[data] >= data.size - args.validation_dataset_size:
            datas_next_start_index[data] = 0
        elif datas_next_start_index[data] >= data.size:
            datas_next_start_index[data] = 0

        if args.verbose:
            epoch_tqdm.write('    {}: Done "Create Training Dataset" in {:.3f} s'.format(
                get_date_str(), time.time() - start))

    # training loop
    train_dataset = train_dataset.shuffle(args.batch_size)
    batch_tqdm = tqdm.tqdm(train_dataset, desc='Training', total=args.train_dataset_size //
                           args.batch_size + 1, leave=False, unit='batch')
    for batch_x, batch_y in batch_tqdm:
        # retrieve batch data
        batch_input = batch_x
        batch_policy = batch_y[src.policy_output_name]
        batch_value = batch_y[src.value_output_name]

        # computing loss and gradients
        with tf.GradientTape() as tape:
            logits_policy, logits_value = model(batch_input, training=True)

            loss_policy_value = loss_policy_fn(batch_policy, logits_policy)
            loss_value_value = loss_value_fn(batch_value, logits_value)

            loss_value = loss_policy_value + loss_value_value

        gradients = tape.gradient(loss_value, model.trainable_weights)

        # apply gradient descent
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # update metrics
        epoch_train_loss(loss_value)
        epoch_train_loss_policy(loss_policy_value)
        epoch_train_loss_value(loss_value_value)
        epoch_train_accuracy_policy(batch_policy, logits_policy)
        epoch_train_accuracy_value(batch_value, logits_value)

    # validation loop
    validation_tqdm = tqdm.tqdm(validation_dataset, desc='Validation',
                                total=args.validation_dataset_size // args.batch_size + 1, leave=False, unit='batch')
    for validation_x, validation_y in validation_tqdm:
        # retrive validation data
        validation_input = validation_x
        validation_policy = validation_y[src.policy_output_name]
        validation_value = validation_y[src.value_output_name]

        # compute logits
        logits_policy, logits_value = model(validation_input)

        # compute losses
        loss_policy_value = loss_policy_fn(validation_policy, logits_policy)
        loss_value_value = loss_value_fn(validation_value, logits_value)
        loss_value = loss_policy_value + loss_value_value

        # update metrics
        epoch_validation_loss(loss_value)
        epoch_validation_loss_policy(loss_policy_value)
        epoch_validation_loss_value(loss_value_value)
        epoch_validation_accuracy_policy(validation_policy, logits_policy)
        epoch_validation_accuracy_value(validation_value, logits_value)

    # updates logs
    epoch_end_time = time.time()
    train_logs.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(
        epoch,
        epoch_start_time,
        epoch_end_time,
        data.name,
        epoch_start_index,
        epoch_train_loss.result(),
        epoch_train_loss_policy.result(),
        epoch_train_loss_value.result(),
        epoch_train_accuracy_policy.result(),
        epoch_train_accuracy_value.result(),
        epoch_validation_loss.result(),
        epoch_validation_loss_policy.result(),
        epoch_validation_loss_value.result(),
        epoch_validation_accuracy_policy.result(),
        epoch_validation_accuracy_value.result(),
    ))

    if args.verbose:
        epoch_tqdm.write('    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}\n    {}={}'.format(
            src.TrainLogsMetrics.train_loss, epoch_train_loss.result(),
            src.TrainLogsMetrics.train_loss_policy, epoch_train_loss_policy.result(),
            src.TrainLogsMetrics.train_loss_value, epoch_train_loss_value.result(),
            src.TrainLogsMetrics.train_accuracy_policy, epoch_train_accuracy_policy.result(),
            src.TrainLogsMetrics.train_accuracy_value, epoch_train_accuracy_value.result(),
            src.TrainLogsMetrics.validation_loss, epoch_validation_loss.result(),
            src.TrainLogsMetrics.validation_loss_policy, epoch_validation_loss_policy.result(),
            src.TrainLogsMetrics.validation_loss_value, epoch_validation_loss_value.result(),
            src.TrainLogsMetrics.validation_accuracy_policy, epoch_validation_accuracy_policy.result(),
            src.TrainLogsMetrics.validation_accuracy_value, epoch_validation_accuracy_value.result(),
        ))

    if args.backup:
        if best_accuracy_policy is None:
            best_accuracy_policy = epoch_validation_accuracy_policy.result()
            best_accuracy_value = epoch_validation_accuracy_value.result()

            model.save('{}best_accuracy_policy.h5'.format(backup_dir))
            model.save('{}best_accuracy_value.h5'.format(backup_dir))

        if best_accuracy_policy < epoch_validation_accuracy_policy.result():
            best_accuracy_policy = epoch_validation_accuracy_policy.result()
            model.save('{}best_accuracy_policy.h5'.format(backup_dir))

        if best_accuracy_value < epoch_validation_accuracy_value.result():
            best_accuracy_value = epoch_validation_accuracy_value.result()
            model.save('{}best_accuracy_value.h5'.format(backup_dir))

        model.save('{}last_model.h5'.format(backup_dir))


# save model
if args.verbose:
    print('{}: Begin "Save model"'.format(get_date_str()))
    start = time.time()

src.create_dir(src.models_dir)
model.save(args.model_file)

if args.verbose:
    print('{}: Done "Save model" in {:.3f} s'.format(
        get_date_str(), time.time() - start))

# closing operations
if args.verbose:
    print('{}: Begin "Shutdown"'.format(get_date_str()))
    start = time.time()

train_logs.close()

if args.verbose:
    print('{}: Done "Shutdown" in {:.3f} s'.format(
        get_date_str(), time.time() - start))
