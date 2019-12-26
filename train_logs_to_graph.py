import argparse

import pandas as pd

import matplotlib.pyplot as plt
import src

parser = argparse.ArgumentParser()
parser.add_argument('logs', help='path to train logs')
parser.add_argument(
    '--output', '-o', default='{}train_graph.png'.format(src.results_dir), help='output path')

args = parser.parse_args()

src.create_dir(src.results_dir)

df = pd.read_csv(args.logs, sep=';')

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
axes = [ax1, ax2, ax3, ax4]

ax_data = fig.add_subplot(gs[1, :])

x = df[src.train_logs_epoch]

axes[0].plot(x, df[src.TrainLogsMetrics.train_loss],
             alpha=0.7, label=src.TrainLogsMetrics.train_loss)
axes[0].plot(x, df[src.TrainLogsMetrics.validation_loss],
             alpha=0.7, label=src.TrainLogsMetrics.validation_loss)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend()

axes[1].plot(x, df[src.TrainLogsMetrics.train_loss_policy],
             alpha=0.7, label=src.TrainLogsMetrics.train_loss_policy)
axes[1].plot(x, df[src.TrainLogsMetrics.validation_loss_policy],
             alpha=0.7, label=src.TrainLogsMetrics.validation_loss_policy)
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss policy')
axes[1].legend()

axes[2].plot(x, df[src.TrainLogsMetrics.train_loss_value],
             alpha=0.7, label=src.TrainLogsMetrics.train_loss_value)
axes[2].plot(x, df[src.TrainLogsMetrics.validation_loss_value],
             alpha=0.7, label=src.TrainLogsMetrics.validation_loss_value)
axes[2].set_xlabel('epoch')
axes[2].set_ylabel('loss value')
axes[2].legend()

axes[3].plot(x, df[src.TrainLogsMetrics.train_accuracy_policy],
             alpha=0.7, label=src.TrainLogsMetrics.train_accuracy_policy)
axes[3].plot(x, df[src.TrainLogsMetrics.validation_accuracy_policy],
             alpha=0.7, label=src.TrainLogsMetrics.validation_accuracy_policy)
axes[3].set_xlabel('epoch')
axes[3].set_ylabel('accuracy policy')
axes[3].legend()

nb_epoch_per_data = list(df.groupby(src.train_logs_data).count()[
                         src.train_logs_epoch])
datas = list(pd.unique(df[src.train_logs_data]))

bars = ax_data.bar(range(len(datas)), nb_epoch_per_data)

ax_data.set_ylabel('number of epoch')
ax_data.set_xticks(range(len(datas)))
ax_data.set_xticklabels(datas)

fig.tight_layout()
fig.savefig(args.output, dpi=100)