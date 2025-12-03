# -*- coding: utf-8 -*-
### conda code v1

# !pip install biopython
# !pip install folium
# !curl -O https://raw.githubusercontent.com/jperkel/example_notebook/master/NC_005816.gb

##################
import os
import time
import numpy as np
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from regressor_classes import Dataset, Model


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='gcPBM regressor for affinity prediction')
parser.add_argument('--protein', type=str, help='Protein name')
parser.add_argument('--concentration', type=str, help='Concentration label')
parser.add_argument('--data-dir', type=str, default=None,
                    help='Data directory (default: proteins/{protein} or test_data/gcPBM/{protein} for default case)')
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
concentration = args.concentration

if args.data_dir is not None:
    data_dir = args.data_dir
else:
    # Original behaviour: proteins/{protein} unless using the built-in test case
    if len(sys.argv) > 1:
        data_dir = f'proteins/{protein}'
    else:
        data_dir = f'test_data/gcPBM/{protein}'


#####################################
# files required to run:
# '{data_dir}/{protein}_{concentration}.txt'
# '{data_dir}/{protein}_freq_matrix_6.txt'_freq_matrix_6.txt
#####################################

# yeast

raw_data = pd.read_csv(f'{data_dir}/{protein}/{protein}_{concentration}.txt', sep='\t')
proc_data = pd.concat([raw_data[["Sequence", "Intensity"]]])
proc_data = proc_data.dropna()
proc_data = proc_data.reset_index()
strings = list(proc_data["Sequence"])
strings = [seq[3:33] for seq in strings]
values = proc_data["Intensity"]

min_val = min(values)
normalization = max(values) - min(values)
values = list((values - min_val) / normalization)

dictionary = dict(zip(values, strings))

plt.plot(sorted(values))
print(strings[0][12:18])

freq_matrix = pd.read_csv(f'{data_dir}/{protein}/{protein}_freq_matrix_6.txt', sep=r'\s+')
freq_matrix = np.array(freq_matrix)
translate = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
reverse = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

len_aln = len(strings[0])

# scores

l = list(np.arange(0, 12 - 5))
for k in np.arange(18, 30 - 5):
    l.append(k)

discarded = []
for s in np.arange(len(strings)):
    string = strings[s]
    scores = []
    for i in l:
        score = 0
        for j in np.arange(0, len(freq_matrix)):
            a = translate[string[i + j]]
            score = score + freq_matrix[j][a]
        scores.append(score)
    if any(sc > 2.5 for sc in scores):
        discarded.append(s)

discarded = list(discarded)
print('We have discarded', len(discarded) / len(strings) * 100, '% of the data')

# options:

# no smart under, no weighting
df_train = pd.DataFrame({'ID_REF': strings, 'VALUE': values})
df_train = df_train.drop(discarded)
df_train = df_train.reset_index(drop=True)
# df_train = df_train.sample(frac=1).reset_index(drop=True)[0:15000]
len(df_train)
# # smart under, no weighting
# df_train = pd.read_csv(f'{data_dir}/{protein}_training.txt', sep='\t')

# # no smart under, weighting
# df_train = pd.DataFrame({'ID_REF': strings, 'VALUE': values, 'WEIGHT':weights})
# df_train = df_train.sample(frac=1).reset_index(drop=True)[0:15000]


the_features = {0: ['electrostatic'], 1: ['presence_tetramer'], 2: ['avg'], 3: ['diagonal_fce'],
                4: ['presence_tetramer', 'avg', 'diagonal_fce', 'electrostatic'], 5: ['avg', 'diagonal_fce']}
len_aln = len(strings[0])

chosen_features = the_features[4]
model_target = args.model_target
regressor = args.regressor
training_set_size = args.training_set_size
randomize_fce = args.randomize_fce
score = args.score
selected_tetramers = list(np.arange(0, len_aln - 3))

# PyTorch model parameters (only used if regressor is 'DFN')
default_pytorch_params = {
    'layers': [64, 32, 1],  # [hidden1, hidden2, ..., output]
    'lr': 0.001,
    'normalization': True,
    'num_epochs': 100,
    'batch_size': 32
}

if args.config is not None:
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    if 'pytorch_model_params' in cfg:
        cfg = cfg['pytorch_model_params']
    pytorch_model_params = {**default_pytorch_params, **cfg}
else:
    pytorch_model_params = default_pytorch_params

## time

start_time = time.time()
dset_train = Dataset(protein, df_train, model_target, randomize_fce, chosen_features, score, selected_tetramers)
model = Model(dset_train, len_aln, regressor, training_set_size,
              pytorch_model_params=pytorch_model_params)
print('the model has been trained')
print("It took %s seconds" % (time.time() - start_time))

fig = plt.gcf()
fig.set_size_inches(5, 5)
model.predict()
print(model.y_test.shape, model.y_pred.shape)
print('The correlation is ', model.r2)
model.plot()
plt.plot([0, 1], color='red')

# with open(f'output/{protein}/electro_correlations.txt','a') as file:
#    file.write("0vs%s\t" % cycle)
#    file.write("%s\n" % model.r2)

# Only process feature importances if available (sklearn models)
if model.features and len(model.features) > 0:
    df = pd.DataFrame(model.features, columns=['Feature', 'Importance'])
    
    index = [k for k in range(len(df)) if df['Feature'].iloc[k] == 'Presence']
    p = sum([df['Importance'].iloc[k] for k in index]) if index else 0
    
    index = [k for k in range(len(df)) if df['Feature'].iloc[k] == 'Electro']
    e = sum([df['Importance'].iloc[k] for k in index]) if index else 0
else:
    # If no feature importances available (e.g., for non-tree models), skip this analysis
    print("Feature importance analysis not available for this regressor type")
    p, e = 0, 0

if model.features and len(model.features) > 0:
    shape = 1 - e - p
    
    y = np.array([p, e, shape])
    
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    labels = ["Presence", "Electrostatic", "Shape"]
    patches, texts = plt.pie(y, startangle=0)
    plt.legend(patches, labels, loc="best")
    # plt.pie(y)
    plt.show()
    
    plt.xlabel('Feature number')
    plt.ylabel('Relative importance (%)')
    plt.legend('Top 30 features')
    if model.l and len(model.l) > 0:
        plt.bar(range(len(model.l[0:10])), model.l[0:10], color='red', align="center", )
    
    print(df.head(10))
if not os.path.isdir('output_gcpbm'):
    os.mkdir('output_gcpbm')
if not os.path.isdir(f'output_gcpbm/{protein}'):
    os.mkdir(f'output_gcpbm/{protein}')

# save the model to disk
filename = f'output_gcpbm/{protein}/model.pck'
if regressor in ['DFN']:
    import torch
    torch.save(model.regressor.state_dict(), filename.replace('.pck', '.pth'))
    # Also save the full model object for compatibility
    pickle.dump(model, open(filename, 'wb'))
else:
    pickle.dump(model, open(filename, 'wb'))
