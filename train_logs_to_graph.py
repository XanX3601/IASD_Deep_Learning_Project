import src
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('logs', help='path to train logs')
parser.add_argument('--output', '-o', default='{}train_graph.png'.format(src.results_dir), help='output path')

args = parser.parse_args()

src.create_dir(src.results_dir)

df = pd.read_csv(args.logs, sep=';')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

x = df[src.train_logs_epoch]

axes[0].plot(x, df[src.TrainLogsMetrics.train_loss], alpha=0.7, label=src.TrainLogsMetrics.train_loss)
axes[0].plot(x, df[src.TrainLogsMetrics.validation_loss], alpha=0.7, label=src.TrainLogsMetrics.validation_loss)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend()

axes[1].plot(x, df[src.TrainLogsMetrics.train_loss_policy], alpha=0.7, label=src.TrainLogsMetrics.train_loss_policy)
axes[1].plot(x, df[src.TrainLogsMetrics.validation_loss_policy], alpha=0.7, label=src.TrainLogsMetrics.validation_loss_policy)
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss policy')
axes[1].legend()

axes[2].plot(x, df[src.TrainLogsMetrics.train_loss_value], alpha=0.7, label=src.TrainLogsMetrics.train_loss_value)
axes[2].plot(x, df[src.TrainLogsMetrics.validation_loss_value], alpha=0.7, label=src.TrainLogsMetrics.validation_loss_value)
axes[2].set_xlabel('epoch')
axes[2].set_ylabel('loss value')
axes[2].legend()

axes[3].plot(x, df[src.TrainLogsMetrics.train_accuracy_policy], alpha=0.7, label=src.TrainLogsMetrics.train_accuracy_policy)
axes[3].plot(x, df[src.TrainLogsMetrics.validation_accuracy_policy], alpha=0.7, label=src.TrainLogsMetrics.validation_accuracy_policy)
axes[3].set_xlabel('epoch')
axes[3].set_ylabel('accuracy policy')
axes[3].legend()

fig.tight_layout()
fig.savefig(args.output, dpi=100)
