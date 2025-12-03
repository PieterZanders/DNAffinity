# -*- coding: utf-8 -*-
# !pip install biopython
# !pip install folium
# !curl -O https://raw.githubusercontent.com/jperkel/example_notebook/master/NC_005816.gb

import sys, os
import argparse
import json
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from collections import defaultdict
from itertools import product

from regressor_classes import *

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='SELEX regressor for affinity prediction')
parser.add_argument('--protein', type=str, help='Protein name')
parser.add_argument('--cycle', type=str, help='Cycle number')
parser.add_argument('--data-dir', type=str, default=None,
                    help='Data directory (default: test_data/{protein})')
parser.add_argument('--regressor', type=str, default='DFN',
                    choices=['random_forest', 'linear', 'ridge', 'svr',
                             'DFN'],
                    help='Regressor type (default: DFN)')
parser.add_argument('--training-set-size', type=float, default=0.9,
                    help='Training set size fraction (default: 0.9)')
parser.add_argument('--randomize-fce', action='store_true',
                    help='Randomize FCE features')
parser.add_argument('--score', type=str, default='Median_intensity',
                    help='Score column name (default: Median_intensity)')
parser.add_argument('--model-target', type=str, default='octamers',
                    choices=['octamers', 'tetramers'],
                    help='Model target (default: octamers)')
parser.add_argument('--config', type=str, default=None,
                    help='Path to JSON file with PyTorch model parameters')

args = parser.parse_args()

protein = args.protein
cycle = args.cycle
data_dir = args.data_dir if args.data_dir else f'test_data/{protein}'

#####################################
# files required to run:
# '{data_dir}/{cycle}.txt'
#####################################

deca = pd.read_csv(f'{data_dir}/{protein}/{cycle}.txt', sep='\t')

strings = list(deca["Kmer"])
values = list(deca["Affinity"])
# counts = list(deca["ExpectedCount"])
counts = list(deca["ObservedCount"])
prob = list(deca["Probability"])

# undersampling

prob_bound = sorted(prob)[int(len(prob)*0.9)]

mat = []
for i in np.arange(0, len(prob)):
    if prob[i] > prob_bound:
        mat.append([strings[i], values[i]])

# ordered

if not os.path.isdir('output_selex'):
    os.mkdir('output_selex')
if not os.path.isdir(f'output_selex/{protein}'):
    os.mkdir(f'output_selex/{protein}')


with open(f'output_selex/{protein}/{protein}_training_ordered.txt', 'w') as file:
    file.write('ID_REF\tVALUE\n')
    for vector in mat:
        file.write("%s\t" % vector[0])
        file.write("%s\n" % vector[1])

# writes the randomized file ready for training

np.random.shuffle(mat)

with open(f'output_selex/{protein}/SELEX_training.txt', 'w') as file:
    file.write('ID_REF\tVALUE\n')
    for vector in mat:
        file.write("%s\t" % vector[0])
        file.write("%s\n" % vector[1])


the_features = {0: ['diagonal_fce'], 1: ['presence_tetramer'], 2: ['avg'],
                3: ['presence_tetramer', 'avg', 'diagonal_fce'],
                4: ['presence_tetramer', 'avg', 'diagonal_fce', 'electrostatic'],
                5: ['avg', 'diagonal_fce']}
len_aln = len(strings[0])

df_train = pd.read_csv(f'output_selex/{protein}/SELEX_training.txt', sep='\t')

chosen_features = the_features[3]
model_target = args.model_target
regressor = args.regressor
training_set_size = args.training_set_size
randomize_fce = args.randomize_fce
score = args.score
selected_tetramers = list(np.arange(0, len_aln-3))

# PyTorch model parameters (only used if regressor is 'DFN')
default_pytorch_params = {
    'layers': [256, 128, 64, 32, 1],  # [hidden1, hidden2, ..., output]
    'lr': 1e-4,
    'normalization': True,
    'num_epochs': 100,
    'batch_size': 16
}

if args.config is not None:
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    if 'pytorch_model_params' in cfg:
        cfg = cfg['pytorch_model_params']
    pytorch_model_params = {**default_pytorch_params, **cfg}
else:
    pytorch_model_params = default_pytorch_params

# time
start_time = time.time()
dset_train = Dataset(protein, df_train, model_target, randomize_fce, chosen_features, score, selected_tetramers)
model = Model(dset_train, len_aln, regressor, training_set_size, pytorch_model_params=pytorch_model_params)
print('the SELEX model has been trained')
print("It took %s seconds" % (time.time() - start_time))

model.predict()
print(model.y_test.shape, model.y_pred.shape)
print('The r2 is ', model.r2)

# save the model to disk
if not os.path.isdir('trained_models'):
    os.mkdir('trained_models')
filename = f'trained_models/{protein}_{regressor}_model.sav'
if regressor in ['DFN']:
    import torch
    torch.save(model.regressor.state_dict(), filename.replace('.sav', '.pth'))
    # Also save the full model object for compatibility
    pickle.dump(model, open(filename, 'wb'), protocol=4)
else:
    pickle.dump(model, open(filename, 'wb'), protocol=4)

