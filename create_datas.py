import os

import golois
import tqdm

import src

sgf_directory = src.sgf_directory
data_directory = src.data_directory
game_txt_path = 'game.txt'

if not os.path.exists(sgf_directory):
    print('Please download sgf files using "download_sgf.py" first')
    exit(0)

if not os.path.exists(data_directory):
    os.mkdir(data_directory)

for sgfs in tqdm.tqdm(os.listdir(sgf_directory), desc='Creating data for each model', unit='model'):
    sgfs_path = '{}{}/'.format(sgf_directory, sgfs)

    with open(game_txt_path, 'w') as game_txt:
        for sgf in os.listdir(sgfs_path):
            sgf_path = '{}{}'.format(sgfs_path, sgf)
            game_txt.write('{}\n'.format(sgf_path))

    golois.convert(game_txt_path, '{}{}.data'.format(data_directory, sgfs))

os.remove(game_txt_path)
