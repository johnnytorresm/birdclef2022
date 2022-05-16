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


# In[ ]:


def get_all_labels(bird, filename):
    
    search_key = bird + '/' + filename
    
    primary_label = train_md.loc[train_md['filename'] == search_key]['primary_label'].tolist()
    secondary_labels = train_md.loc[train_md['filename'] == search_key]['secondary_labels'].values
    
    if len(primary_label)==0:
        primary_label = [bird]
        
    if len(secondary_labels)==0:
        secondary_labels = []
    else:
        secondary_labels = ast.literal_eval(secondary_labels[0])
    
    all_labels = primary_label + secondary_labels

    return all_labels
    
print('Functions have been defined')
        


# In[ ]:


# Import all the required libraries.

import numpy as np
import pandas as pd
import seaborn as sns
from numpy import argmax
import ast

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
import tensorflow as tf

# Import necessary libraries for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import random

print('Libraries have been imported')


# In[ ]:


# Reading metadata information
train_md = pd.read_csv('../input/birdclef-2022/train_metadata.csv')
train_md.head()


# In[ ]:


# Selecting the 21 species for the competition
# contained in the file scored_birds.json provided by Kaggle

import json
 
# Opening JSON file
f = open('/kaggle/input/birdclef-2022/scored_birds.json')
 
# returns JSON object as a dictionary
birds = json.load(f)
 
print(birds)


# ### Data augmentation

# In[ ]:


import shutil

print('Data augmentation begins... Copying files...\n')

# Copy the whole input data to /kaggle/working - considered species only
for b in birds:
    source_dir = f"../input/birdclef-2022/train_audio/{b}"
    destination_dir = f"./train_audio/{b}"
    print('Copying files from ', source_dir, ' to ', destination_dir)
    shutil.copytree(source_dir, destination_dir)

print('Files have been copied...')


# In[ ]:


# Counting number of sound files per species

no_files_df = pd.DataFrame(columns=['Species', 'No_Files'])

for b in birds:
    no_files = len([name for name in os.listdir(f'./train_audio/{b}')])
    dict_row = {'Species': b, 'No_Files': no_files}
    no_files_df = no_files_df.append(dict_row, ignore_index = True)
                    
# Some stats from previous counting
# This is useful to know how much we are going to augment data

median_files = (np.median(no_files_df['No_Files']))
no_files_df['No_Missing'] = no_files_df['No_Files'].apply(lambda x : int(x - median_files) if x < median_files else 0)

no_files_df


# In[ ]:


# Media augmentation is used only in species which
# the number of files is below the median

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print('Sound data augmentation started...\n')
print("Begin =", current_time, "\n")

for k in range(0,no_files_df.shape[0]):
    nf = int(no_files_df["No_Missing"].iloc[k])
    if nf < 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        nf = np.abs(nf)
        b = no_files_df["Species"].iloc[k]
        print('Species ', b, ' augmenting ', nf, ' files - ', current_time)
        dirlist = os.listdir(f'./train_audio/{b}')
        for i in range(0, nf):
            nf = np.random.randint(low=0, high=len(dirlist))
            readfile = f'./train_audio/{b}/' + dirlist[nf]
            augfile = 'aug_'+ b + str(i) + '.wav'
            signal, sr = librosa.load(readfile)
            type_of_augmentation = random.randint(1,3)
            if type_of_augmentation==1:
                augmented_signal = librosa.effects.time_stretch(signal, rate=np.random.random())
            elif type_of_augmentation==2:
                augmented_signal = np.roll(signal,int(sr/10))
            else:
                augmented_signal = librosa.effects.pitch_shift(signal,sr,n_steps=random.randint(-5,5))
                
            sf.write(f'./train_audio/{b}/' + augfile, augmented_signal, sr)
                
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print('Sound data augmentation ended...')
print("End =", current_time, "\n")


# In[ ]:


# Header for the datafrane containing all features

header = ['species', 
          'filename',
          'rms',
          'chroma_stft', 
          'chroma_cqt', 
          'chroma_cens',
          'mel_spct'
         ]

for i in range(1, 21):
    header += [f'mfcc{i}']

header += ['spec_cent',
           'spec_bw',
           'spec_con',
           'spec_flt',
           'rolloff',
           'tonnetz',
           'zcr'
          ]

f = header[2:]
header += ['label']

birds_fe = pd.DataFrame(columns = header)

for float_field in f:
    birds_fe[float_field] = birds_fe[float_field].astype(float, errors = 'raise')

birds_fe.info(verbose=1)


# In[ ]:


# Browsing all data to extract features. Only scored birds will be considered.
# This processs takes some time to complete

print('Feature extraction process.. takes a while to complete...\n')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Begin =", current_time)

birds_data = []   # Temporary list to containt all features as they are extracted

# for bird in ['afrsil1']:
for bird in birds:   # only selected birds for the competition
# for bird in ['akiapo']:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('Processing species: ', bird, ' at ', current_time)
    
    for filename in os.listdir(f'/kaggle/input/birdclef-2022/train_audio/{bird}'):
    
        # Feature extraction
        songname = f'/kaggle/input/birdclef-2022/train_audio/{bird}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=60)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        mel_spct = librosa.feature.melspectrogram(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flt = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        new_row = [bird, 
                   filename, 
                   np.mean(rms),
                   np.mean(chroma_stft),
                   np.mean(chroma_cqt),
                   np.mean(chroma_cens),
                   np.mean(mel_spct)
                  ]
        
        for e in mfcc:
            new_row.append(np.mean(e))
            
        new_row.append(np.mean(spec_cent))
        new_row.append(np.mean(spec_bw))
        new_row.append(np.mean(spec_con))
        new_row.append(np.mean(spec_flt))
        new_row.append(np.mean(rolloff))
        new_row.append(np.mean(tonnetz))
        new_row.append(np.mean(zcr))        
        
        all_labels = get_all_labels(bird, filename)

        for i in range(0, len(all_labels)):
            final_row = new_row.copy()
            final_row.append(all_labels[i])
            birds_data.append(final_row)            

# Creating the dataframe containing the extracted features
birds_df = pd.DataFrame(birds_data, columns = header)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End =", current_time)

print('\nFeatures extracted and saved to disk...\n')

birds_df.head()


# In[ ]:


# Checking the correlation between columns
          
sns.heatmap(birds_df.corr());


# In[ ]:


# Dropping features
if 'species' in birds_df.columns:
    birds_df.drop(columns=['species'], inplace=True)
if 'filename' in birds_df.columns:
    birds_df.drop(columns=['filename'], inplace=True)

birds_df.head()


# In[ ]:


# Encoding the Labels
birds_list = birds_df.iloc[:, -1]
encoder = LabelEncoder()

# Fittng the data
y = encoder.fit_transform(birds_list)

# Dividing data into training and Testing set
X = birds_df.iloc[:, :-1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15)

# Scaling the Feature columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[ ]:


keras.backend.clear_session()


# In[ ]:


#  Building an ANN model.

model = Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.25))
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
                       epochs=50,
                       validation_data=(X_val, y_val),
                       batch_size=16,
                       verbose=1
                      )

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End =", current_time)

print('\nModel fitted')


# In[ ]:


# Plotting the results
from matplotlib import pyplot

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

# predicting labels
y_pred = model.predict(X_test)

# Index of top values
y_pred = tf.argmax(y_pred, axis=1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred, average='micro')
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred, average='micro')
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred, average='micro')
print('F1 score: %f' % f1)


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
        chroma_cqt = librosa.feature.chroma_cqt(y=chunk, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=chunk, sr=sr)
        mel_spct = librosa.feature.melspectrogram(y=chunk, sr=sr)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
        spec_con = librosa.feature.spectral_contrast(y=chunk, sr=sr)
        spec_flt = librosa.feature.spectral_flatness(y=chunk)
        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=chunk, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(chunk)


        chunk_features = [np.mean(rms),
                          np.mean(chroma_stft),
                          np.mean(chroma_cqt),
                          np.mean(chroma_cens),
                          np.mean(mel_spct)
                         ]
        
        for e in mfcc:
            chunk_features.append(np.mean(e))
            
        chunk_features.append(np.mean(spec_cent))
        chunk_features.append(np.mean(spec_bw))
        chunk_features.append(np.mean(spec_con))
        chunk_features.append(np.mean(spec_flt))
        chunk_features.append(np.mean(rolloff))
        chunk_features.append(np.mean(tonnetz))
        chunk_features.append(np.mean(zcr)) 

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

