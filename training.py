# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### imports and a couple useful functions:

# +
from __future__ import print_function, division

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from tqdm import tqdm
from glob import glob
import datetime
import os, sys

pd.options.display.max_rows = 30
pd.options.display.max_columns = 40

from collections import OrderedDict

## some useful team data in here (converters from short to long):
from basketball_reference_web_scraper import data

years = np.arange(1950, 2019)

## custom printing for my Keras training:
class PrintCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 25 == 0:
            print("epoch {}:  loss = {:.2f}, test loss = {:.2f}".format(
                epoch, logs['loss'], logs['val_loss']))
    def on_train_begin(self, logs={}):
        print("Beginning training...")
    
    def on_train_end(self, logs):
        print("Training completed")

early_stopper = keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=1)
nan_stopper = keras.callbacks.TerminateOnNaN()
        
def find_player_id(player_name, df):
    possible_pids = np.unique(df.index[df['player_name'] == player_name])
    if possible_pids.size == 1:
        return possible_pids[0]
    elif possible_pids.size > 1:
        print("Many options; returning most average points:")
        vals = []
        for pid in possible_pids:
            rows = df.loc[df.index==pid]
            mean_points = np.average(rows['points'])
            vals.append(mean_points)
            print(f'\t{pid} with an average point/year of {mean_points}'.format(
                row['PlayerID'], row['CareerValue']))
        return possible_pids[np.argmax(vals)]
    else:
        print("No exact name matches... possible names:")
        from fuzzywuzzy import process
        matches = process.extract(player_name, np.unique(df['player_name'].values), limit=10)
        for poss in matches:
            print("\t",poss[0])
        print("Returning best match, but maybe incorrect...")
        return find_player_id(matches[0][0], df)
    
class CyclicList(list):
    def __getitem__(self, index):
        return super().__getitem__(index%len(self))


# + {"heading_collapsed": true, "cell_type": "markdown"}
# ### functions that implement recommendations on network size:

# + {"hidden": true}
def recommended_max_hidden_neurons_hobbs(training_sample_size, 
                                   num_input_neurons, 
                                   num_output_neurons,
                                  alpha=2.5):
    """
    recommend the max number of hidden neurons based on I/O & sample size
    
    this recommendation is taken from the second answer (from @hobbs) on
    stackexchange here:  

    https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    """
    
    bottom = alpha * (num_input_neurons + num_output_neurons)
    return training_sample_size/bottom

def recommend_max_hidden_neurons_heaton(num_input_neurons,
                                       num_output_neurons):
    """
    following the third answer, from @jj_, who quotes Heaton,
    we have three rules of thumb:

    * The number of hidden neurons should be between the size 
      of the input layer and the size of the output layer.
    
    * The number of hidden neurons should be 2/3 the size of 
      the input layer, plus the size of the output layer.
    
    * The number of hidden neurons should be less than twice 
      the size of the input layer.    
    """
    
    ## rule one:
    max_size = max([num_input_neurons, num_output_neurons])
    min_size = min([num_input_neurons, num_output_neurons])
    print(f"\tRule one recommends {min_size} - {max_size}")
    
    ## rule two:
    size = np.rint(2*num_input_neurons/3) + num_output_neurons
    print(f"\tRule two recommends {size}", end='')
    if min_size < size < max_size:
        print(", which also satisfies rule 1")
    else:
        print(", which is in conflict with rule 1")
    
    ## rule three:
    max_size_two = 2*num_input_neurons
    print(f"\tRule three recommends no more than {2*max_size_two}")


# -

# ## extract test/training/validate and make recommendations:

# + {"code_folding": []}
def build_training_dataframe(yearly_df, player_df, 
                             training_columns, target_column, 
                             years_to_train_on=2, frac_train=0.75,
                             min_career_length=3, 
                             sort_before_splitting=True,
                             split_randomly=True,
                             drop_pre_1973=True):
    
    assert target_column not in training_columns
    
    msk = player_df['career_length']>=min_career_length
    if drop_pre_1973:
        msk *= player_df['rookie_end_year'] >= 1973
        
    player_subset = player_df.loc[msk]
    
    input_data = []
    
    ## loop over players that meet my requirements
    for pid in player_subset.index:
        ## grab the rows correspoding to that player
        rows = yearly_df.loc[pid]
        
        ## create a dictionary for each player
        pdata = dict(player_id=pid)
        
        ## add the data for the first n years (where n = years_to_train_on) 
        ## of that players career to their dictionary
        for ii in range(years_to_train_on):
            for k in training_columns:
                pdata[k+f'.y{ii+1}'] = rows[k].iloc[ii]
                
        input_data.append(pdata)
    
    ## now turn that dictionary back into a dataframe
    input_data = pd.DataFrame(input_data)
    input_data.set_index('player_id', inplace=True)

    ## and pull the targets out of our original dataset
    target_data = player_subset[target_column]
    if sort_before_splitting:
        input_data[target_column] = target_data
        input_data.sort_values(target_column, inplace=True)
        target_data = input_data.pop(target_column)
    
    total_sample_size = len(target_data)
    all_indices = np.arange(total_sample_size)

    if split_randomly:
        ntrain = int(np.ceil(frac_train*total_sample_size))
        ntest = int(np.ceil((1-frac_train)*total_sample_size/2))
        nvalidate = int(np.ceil((1-frac_train)*total_sample_size/2))

        while ntest + ntrain + nvalidate > total_sample_size:
            ntest -= 1        
        
        train_indices = np.random.choice(all_indices, size=ntrain, replace=False)
        all_indices = np.setdiff1d(all_indices, train_indices)
        
        test_indices = np.random.choice(all_indices, size=ntest, replace=False)
        all_indices = np.setdiff1d(all_indices, test_indices)
        
        validate_indices = np.array(all_indices, copy=True)
    else:
        tt_stride = int(2/(1-frac_train))
        
        test_indices = all_indices[tt_stride//2::tt_stride]
        validate_indices = test_indices + 1
        train_indices = np.setdiff1d(all_indices, np.concatenate((test_indices, validate_indices)))
    
    trainX = input_data.iloc[train_indices]
    trainY = target_data.iloc[train_indices]
    
    testX = input_data.iloc[test_indices]
    testY = target_data.iloc[test_indices]
    
    validateX = input_data.iloc[validate_indices]
    validateY = target_data.iloc[validate_indices]
    
    return trainX, trainY, testX, testY, validateX, validateY
        


# + {"code_folding": [0]}
def extract_and_recommend(yearly_df, player_df, training_columns, target_column, **kwargs):
    trainX, trainY, testX, testY, validateX, validateY = build_training_dataframe(
        yearly_df, player_df, training_columns, target_column, **kwargs)
    
    print("Training on {} columns, so using that many input neurons".format(trainX.shape[1]))
    print("Predicting one column, so using that many output neurons")
    
    input_neurons = trainX.shape[1]
    output_neurons = 1
    
    print("@hobbs recommends {} hidden neurons max".format(
        recommended_max_hidden_neurons_hobbs(len(trainX), input_neurons, output_neurons)))
    print("Heaton recommends:")
    recommend_max_hidden_neurons_heaton(input_neurons, output_neurons)
    
    return trainX, trainY, testX, testY, validateX, validateY


# -

# ### Build and train an model, optionally with hidden layers

def build_and_train(trainX, trainY, 
                    testX, testY, 
                    hidden_layers=None,
                    hidden_layer_neurons=None, hidden_layer_kwargs=None,
                    clear=True,
                    optimizer='Adadelta', loss='mean_squared_error',
                    epochs=250, metrics=[], shuffle=True, batch_size=None,
                    input_layer=None, input_layer_neurons=None, input_layer_kwargs=dict(),
                    output_layer=None, output_layer_neurons=1, output_layer_kwargs=dict()):

    if clear:
        print("Clearing TensorFlow graph")
        keras.backend.clear_session()
    
    model = keras.Sequential()
    
    ## add our input layer:
    if input_layer is not None:
        model.add(input_layer)
    else:
        if input_layer_neurons is None:  input_layer_neurons = trainX.shape[1]
        model.add(keras.layers.Dense(input_layer_neurons, input_shape=[trainX.shape[1]], **input_layer_kwargs))

    ## add any hidden layers
    if hidden_layers is not None:
        ## did we pass in pre-built layers?
        for layer in hidden_layers:
            model.add(layer)
    else:
        for neurons, kwargs in zip(hidden_layer_neurons, hidden_layer_kwargs):
            ## otherwise, assume all are dense
            model.add(keras.layers.Dense(neurons, **kwargs))
        
    ## add our output layer
    if output_layer is not None:
        model.add(output_layer)
    else:
        model.add(keras.layers.Dense(output_layer_neurons, **output_layer_kwargs))

    ## compile our model with our callbacks:
    cblist = [early_stopper, nan_stopper, PrintCallback()]    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    ## and fit it!
    history = model.fit(trainX, trainY, validation_data=(testX, testY),
                        epochs=epochs, verbose=0, callbacks=cblist,
                        shuffle=shuffle, batch_size=batch_size)
    
    return model, history


# ### Functions to check model performance:

# +
def plot_history(history, skip=10, logy=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    keys = [k for k in history.history.keys() if not k.startswith('val_')]
    for key in keys:
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        if logy:
            ax.set_yscale('log')
        ax.plot(hist['epoch'][skip:], hist[key][skip:],
               label='Train Error')
        ax.plot(hist['epoch'][skip:], hist['val_'+key][skip:],
               label = 'Test Error')
        ax.legend()
        
def calculate_mse(model, validateX, validateY):
    prediction = model.predict(validateX).flatten()
    mse = np.mean(np.square(validateY.values - prediction))
    return mse

def plot_pred_vs_actual(model, Xlist, Ylist, labels=None, logaxes=''):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel('actual')
    ax.set_ylabel('prediction')
    if 'x' in logaxes:  ax.set_xscale('log')
    if 'y' in logaxes:  ax.set_yscale('log')
    
    if labels is None:
        labels = ['_nolegend_']*len(Xlist)
    for (x, y, l) in zip(Xlist, Ylist, labels):
        pred = model.predict(x)
        actual = y.values
        
        ax.scatter(actual, pred, label=l, s=25, edgecolor=None)
    ax.legend()    
    return fig, ax


# -

# # OK, moment of truth here -- all the machinery looks set up, so let's do some model building:

# ### read in the data:

yearly_df = pd.read_hdf('scraped/all_years_combined.hdf5', 'nba_stats')
player_df = pd.read_hdf('scraped/all_years_combined.hdf5', 'player_list')

# ### ...and build a model!

# #### Values I'm set up to train for (career-averaged/medianed/maxed/etc values)

player_df.keys()

# #### Values I'm set up to train on (trains on the first N years worth of data for these columns):

count = 0
for k in yearly_df.keys():
    if '.playoffs' in k:
        continue
    print(k.ljust(50), end='')
    if count % 2 == 0:
        print()
    else:
        print(' | ', end='')
    count += 1

# ##### First, a model based on most of the basic stats from the first two years, looking to predict the mean combined vorp:

trainX, trainY, testX, testY, validateX, validateY = extract_and_recommend(yearly_df, player_df,
    ['field_goal_percent', 'three_point_percent', 'attempted_free_throws', 
     'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers'], 
    'vorp.combined-mean')

# +
model, history = build_and_train(trainX, trainY, testX, testY, 
        hidden_layer_neurons=[4, 4, 4, 4], 
        hidden_layer_kwargs=[dict(activation='relu')]*4, 
        loss='mean_absolute_error', 
        metrics=['mean_absolute_percentage_error'],
        clear=True)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])
# -

# What if I add more layers or change up the activations?

# +
model, history = build_and_train(trainX, trainY, testX, testY, 
        hidden_layer_neurons=[8]*4, 
        hidden_layer_kwargs=[dict(activation='elu'), 
                             dict(activation='elu'), 
                             dict(activation='elu'), 
                             dict(activation='elu')], 
        loss='mean_absolute_error', 
        metrics=['mean_absolute_percentage_error'],
        clear=True)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])

# +
model, history = build_and_train(trainX, trainY, testX, testY, 
        hidden_layer_neurons=[12]*3, 
        hidden_layer_kwargs=[dict(activation='relu')]*3, 
        loss='mean_absolute_error', 
        metrics=['mean_absolute_percentage_error'],
        clear=True)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])
# -

# #### Let's try training on advanced stats from the first couple years:

trainX, trainY, testX, testY, validateX, validateY = extract_and_recommend(yearly_df, player_df,
    ['player_efficiency_rating', 'true_shooting_percent', 'vorp'], 'vorp-mean')

# +
model, history = build_and_train(trainX, trainY, testX, testY, 
        hidden_layer_neurons=[8]*4, 
        hidden_layer_kwargs=[dict(activation='relu')]*4, 
        loss='mean_absolute_error', 
        metrics=['mean_absolute_percentage_error'],
        optimizer='RMSprop',
        clear=True)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])

# +
model, history = build_and_train(trainX, trainY, testX, testY, 
        hidden_layer_neurons=[12]*6, 
        hidden_layer_kwargs=[dict(activation='relu'), dict(activation='selu')]*3, 
        loss='mean_absolute_error', 
        metrics=['mean_absolute_percentage_error'],
        optimizer='RMSprop',
        clear=True)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], 
                    [trainY, testY, validateY], 
                    ['train', 'test', 'validate'])

# +
hidden_layers = [keras.layers.Dense(128, activation='relu'), keras.layers.Dense(128, activation='linear'), keras.layers.Dropout(rate=0.5), 
                 keras.layers.Dense(64, activation='relu'), keras.layers.Dense(64, activation='linear'), keras.layers.Dropout(rate=0.5),
                 keras.layers.Dense(32, activation='relu'), keras.layers.Dense(32, activation='linear'), keras.layers.Dropout(rate=0.25), 
                 keras.layers.Dense(16, activation='relu'), keras.layers.Dense(16, activation='linear'), keras.layers.Dropout(rate=0.125)]

model, history = build_and_train(trainX, trainY, testX, testY,  
                                 hidden_layers=hidden_layers, 
                                 loss='mean_absolute_error', optimizer='adam' 
                                 clear=True, batch_size=24)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])
# -

# #### What if I randomly sample, rather than making sure I have high points in my test/train?

trainX, trainY, testX, testY, validateX, validateY = extract_and_recommend(yearly_df, player_df,
    ['player_efficiency_rating', 'true_shooting_percent', 'vorp', 'total_box_plus_minus'], 
    'vorp-mean', sort_before_splitting=False, frac_train=0.8, split_randomly=True)

# +
hidden_layers = [
     keras.layers.Dense(4, activation='relu'),# keras.layers.Dense(128, activation='linear'), keras.layers.Dropout(rate=0.5), 
     keras.layers.Dense(4, activation='relu'),# keras.layers.Dense(64, activation='linear'), keras.layers.Dropout(rate=0.5),
     keras.layers.Dense(4, activation='relu'),# keras.layers.Dense(32, activation='linear'), keras.layers.Dropout(rate=0.25), 
     keras.layers.Dense(4, activation='relu'),# keras.layers.Dense(16, activation='linear'), keras.layers.Dropout(rate=0.125)
]

model, history = build_and_train(trainX, trainY, testX, testY,  
                                 hidden_layers=hidden_layers, 
                                 loss='mse', optimizer='RMSprop',
                                 clear=True, batch_size=8)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])
# -

# #### What if I go hog-wild and train on literally every stat I have?

yearly_df

trainX, trainY, testX, testY, validateX, validateY = extract_and_recommend(yearly_df, player_df,
    set(yearly_df.keys()) - {'player_name', 'positions', 'team', 'primary_position'}, 
   'vorp-mean', sort_before_splitting=True, frac_train=0.8, split_randomly=False)

for df in [trainX, testX, validateX]:
    df.fillna(value=0, inplace=True)

for col in trainX:
#     if type(trainX[col]) == str:
    v = trainX[col][0]
    if isinstance(v, str):
        print(col, type(trainX[col][0]))

# +
hidden_layers = [
#      keras.layers.Dense(128, activation='relu'), keras.layers.Dense(128, activation='linear'), keras.layers.Dropout(rate=0.5), 
#      keras.layers.Dense(128, activation='relu'), keras.layers.Dense(128, activation='linear'), keras.layers.Dropout(rate=0.5), 
#      keras.layers.Dense(64, activation='relu'), keras.layers.Dense(64, activation='linear'), keras.layers.Dropout(rate=0.5),
#      keras.layers.Dense(32, activation='relu'), keras.layers.Dense(32, activation='linear'), keras.layers.Dropout(rate=0.25), 
#      keras.layers.Dense(16, activation='relu'), keras.layers.Dense(16, activation='linear'), keras.layers.Dropout(rate=0.125)
]

model, history = build_and_train(trainX, trainY, testX, testY,  
                                 hidden_layers=hidden_layers, 
                                 loss='mse', optimizer='RMSprop',
                                 clear=True, batch_size=8)

plot_history(history)
print("MSE on validation data is {}".format(calculate_mse(model, validateX, validateY)))
plot_pred_vs_actual(model, [trainX, testX, validateX], [trainY, testY, validateY], ['train', 'test', 'validate'])
# -

# ## What if, instead of doing a regression to predict VORP, I classify players as good or not and see if I can predict that?


