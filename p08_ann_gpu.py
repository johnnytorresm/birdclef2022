#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing libraries

# In[ ]:


# Import all the required libraries.

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import argmax

import librosa
import librosa.display
import os
import os.path
from os import path

import IPython.display as ipd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import random

import soundfile as sf

from PIL import Image
import pathlib
import csv

# Keras
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from tensorflow.keras.optimizers import SGD

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import random

print('Libraries have been imported')


# ### Selecting the species for the competition

# In[ ]:


# Selecting the 21 species for the competition
# contained in the file scored_birds.json provided by Kaggle

import json
 
# Opening JSON file
f = open('/kaggle/input/birdclef-2022/scored_birds.json')
 
# returns JSON object as a dictionary
birds = json.load(f)
 
print(birds)


# ### Feature extraction

# In[ ]:


# Header for the datafrane containing all features

header = 'species filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

birds_fe = pd.DataFrame(columns = header)

float_fields = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2',
                'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
               ]

for float_field in float_fields:
    birds_fe[float_field] = birds_fe[float_field].astype(float, errors = 'raise')

# Browsing all data to extract features. Only scored birds will be considered.
# This processs takes some time to complete

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Begin =", current_time)

birds_data = []   # Temporary list to containt all features as they are extracted

for bird in birds:   # only selected birds for the competition
    print('Processing species: ', bird)
    
    for filename in os.listdir(f'/kaggle/input/birdclef-2022/train_audio/{bird}'):
    
        # Feature extraction
        songname = f'/kaggle/input/birdclef-2022/train_audio/{bird}/{filename}'
        y, sr = librosa.load(songname, mono=True)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        new_row = [bird, filename, np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
        
        for e in mfcc:
            new_row.append(np.mean(e))
        
        new_row.append(bird)
        birds_data.append(new_row)

# Creating the dataframe containing the extracted features
ann_df = pd.DataFrame(birds_data, columns = header)
ann_df.to_csv(r'/kaggle/working/ann_df.csv', index = False)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End =", current_time)

print('\nFeatures extracted and saved to disk...')


# ## ANN Model

# In[ ]:


# dropping unneccesary columns
ann_df = ann_df.drop(columns=['species', 'filename'], axis=1)
ann_df.head()


# In[ ]:


# Encoding the Labels
genre_list = ann_df.iloc[:, -1]
encoder = LabelEncoder()

# Fittng the data
y = encoder.fit_transform(genre_list)

# Dividing data into training and Testing set
X = ann_df.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling the Feature columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Labels encoded. Train and Test Datasets, created and standardized.')


# In[ ]:


keras.backend.clear_session()


# In[ ]:


#  Building an ANN model.

model = Sequential()

model.add(layers.Dense(4096, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(21, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )

model.summary()


# In[ ]:


# Fit the ANN model

print("Fitting the ANN model...")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Begin =", current_time)

classifier = model.fit(X_train,
                       y_train,
                       epochs=100,
                       validation_data=(X_test, y_test),
                       batch_size=16,
                       verbose=0
                      )

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End =", current_time)

print('\nModel fitted')


# In[ ]:


### Plotting the results


# In[ ]:


# Plotting the results
from matplotlib import pyplot

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(classifier.history['loss'], label='train')
pyplot.plot(classifier.history['val_loss'], label='test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(classifier.history['accuracy'], label='train')
pyplot.plot(classifier.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()


# ### Model evaluation

# In[ ]:


# Model Evaluation

# Evaluate the model on the test data using `evaluate`

print("Evaluate on train and test data")
train_eval = model.evaluate(X_train, y_train, batch_size=16)
results = model.evaluate(X_test, y_test, batch_size=16)
print("train loss: ", train_eval[0], "train accuracy: ", train_eval[1])
print("test loss: ", results[0], " - test acc:", results[1])

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(classifier.history['accuracy']), 'r', label='train_accuracy')
ax.plot(np.sqrt(classifier.history['val_accuracy']), 'b' ,label='val_accuracy')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(classifier.history['loss']), 'r', label='train')
ax.plot(np.sqrt(classifier.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)


# In[ ]:


# Generate predictions (probabilities -- the output of the last layer) --
# on new data using predict

# predict probabilities for test set
print("Generate predictions for X_test")
pred_x = model.predict(X_test, verbose=0)

# predict crisp classes for test set
print("Generate classes for X_test")
classes_x = np.argmax(pred_x,axis=1)

# reduce to 1d array
pred_x = pred_x[:, 0]

print('Predictions done...')


# ### Submit to BirdCLEF2022

# In[ ]:


# Part 1:

# Directory where sound files have been placed
test_audio_dir = '/kaggle/input/birdclef-2022/test_soundscapes/'

# All sound files will be splitted into 5-second chunks
chunk_size = 5   

# Getting all the file names from directory
file_list = [f.split('.')[0] for f in sorted(os.listdir(test_audio_dir))]
print('Number of test soundscapes found:', len(file_list))

# extracted features names
header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()

# This is where results are stored before writing the submission file
dict_pred = {'row_id': [], 'target': []}

# Part 2:

# Traverse all files inside the folder and make chunks of each audio file
for afile in file_list: 
    file_path = test_audio_dir + afile + '.ogg'
    print(f"Making chunks of size {chunk_size}s of file: {afile}")

    # Load the file
    sig, sr = librosa.load(file_path)
    
    # Get number of samples for <chunk_size> seconds
    buffer = chunk_size * sr
    samples_total = len(sig)
    samples_wrote = 0
    
    counter = 1
    
    # each file is chopped up into several chunks.
    # each chunk is preprocessed and its features extracted to make a prediction
    while samples_wrote < samples_total:
        # check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        chunk = sig[samples_wrote : (samples_wrote + buffer)]
        chunk_end_time = counter * 5

        # chunk_features holds all extracted features from the chunk.
        # this file is fully rewritten for each chunk
        
        chunk_features = []

        # Feature extraction from chunk
        rms         = librosa.feature.rms(y=chunk)
        chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr)
        spec_cent   = librosa.feature.spectral_centroid(y=chunk, sr=sr)
        spec_bw     = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
        rolloff     = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
        zcr         = librosa.feature.zero_crossing_rate(y=chunk)
        mfcc        = librosa.feature.mfcc(y=chunk, sr=sr)
        to_append   = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'

        # File is read back to a dataframe

        chunk_features = [np.mean(chroma_stft), 
                          np.mean(rms), 
                          np.mean(spec_cent), 
                          np.mean(spec_bw), 
                          np.mean(rolloff), 
                          np.mean(zcr)
                         ]
        
        for e in mfcc:
            chunk_features.append(np.mean(e))

        # Scaling the features extracted from the chunk
        chunk_scaled = scaler.transform([chunk_features])
        
        # and predicting labels
        chunk_pred = model.predict(chunk_scaled) # ANN model
        
        for bird in birds:
            i = encoder.transform([bird])
            row_id = afile + '_' + bird + '_' + str(chunk_end_time)
            
            # Put the result into our prediction dict and 
            # apply a "confidence" threshold of 0.5
            dict_pred['row_id'].append(row_id)
            dict_pred['target'].append(True if chunk_pred[0][i]>0.5 else False)

        # next chunk
        counter += 1
        samples_wrote += buffer

# Part 03

# All sound files have been now splitted and the chunks, predicted.
# With the resulting dictionary make a new data frame and look at some results

results = pd.DataFrame(dict_pred, columns = ['row_id', 'target'])
    
# Convert results to csv
results.to_csv("/kaggle/working/submission.csv", index=False)

print('Results have been submitted')

