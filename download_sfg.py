import os
import argparse
import tarfile

import pandas as pd
import requests
import tqdm

import src

parser = argparse.ArgumentParser(description='Download the .sgf files')

args = parser.parse_args()

sgf_directory = src.sgf_directory

df = pd.read_csv(
    'https://dl.fbaipublicfiles.com/elfopengo/v2_training_run/urls.csv')
df = df[(df.model_version >= 1466000) & (df.model_version <= 1499000)]

if not os.path.exists(sgf_directory):
    os.mkdir(sgf_directory)

for sgf_url in tqdm.tqdm(df.selfplay_sgf_url, desc='Downloading selfplays for each model version', total=df.shape[0], unit='tar'):
    tar_name = os.path.join('.', sgf_url.split('/')[-1])

    request = requests.get(sgf_url, stream=True, headers={
                           'Accept-Encoding': None})

    file_size = int(request.headers['content-length'])
    chunck_size = 1024
    num_bars = file_size // chunck_size

    iterator = tqdm.tqdm(request.iter_content(chunk_size=chunck_size), total=num_bars,
                         unit='KB', desc='Downloading {}'.format(tar_name), leave=False)

    with open(tar_name, 'wb') as file:
        for chunck in iterator:
            file.write(chunck)

    with tarfile.open(tar_name) as tar:
        tar.extractall(sgf_directory)

    os.remove(tar_name)
