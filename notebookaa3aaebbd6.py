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


# ## Importing libraries

# In[ ]:


# Import all the required libraries.

import numpy as np
import pandas as pd

import librosa
from librosa import display

import os
from os import path

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

# Import necessary libraries for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pathlib
import csv

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

print('Libraries have been imported')


# ### Selecting species for the competition

# In[ ]:


# Selecting the 21 species for the competition
# contained in the file scored_birds.json provided by Kaggle

import json
 
# Opening JSON file
f = open('../input/birdclef-2022/scored_birds.json')
 
# returns JSON object as a dictionary
birds = json.load(f)
 
print(birds)


# ### Feature extraction

# In[ ]:


# Header for the datafrane containing all features

print('Creating dataframe to contain features...\n')

l = ['species', 'filename', 'duration', 'rmse' ]

for f in ['chroma_stft', 'chroma_cqt', 'chroma_cens']:
    for i in range(0,12):
        l += [f+str(i)]
l += ['spec_cent', 'spec_bw']

for i in range(0,7):
        l += ['spec_con' + str(i)]

l += ['spec_flt', 'rolloff', 'zcr']

for i in range(0,6):
        l += ['tonnetz' + str(i)]

for i in range(0,20):
        l += ['mfcc' + str(i)]

f = l[2:]
l += ['label']

birds_fe = pd.DataFrame(columns = l)

for float_field in f:
    birds_fe[float_field] = birds_fe[float_field].astype(float, errors = 'raise')

# Browsing all data directories to extract features
# Only selected birds will be considered
# this processs takes some time to complete

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Begin =", current_time, "\n")

for b in birds:   # only selected birds for the competition
    print('Processing species => ', b)
    for filename in os.listdir(f'../input/birdclef-2022/train_audio/{b}'):

        songname = f'../input/birdclef-2022/train_audio/{b}/{filename}'
        y, sr = librosa.load(songname, mono=True)
        
        # Extract features
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flt = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        new_row = [b, filename, duration]
        
        new_row.append(np.mean(rms))
        
        for k in range(0,12):
            new_row.append(np.mean(chroma_stft[k,:]))
        
        for k in range(0,12):
            new_row.append(np.mean(chroma_cqt[k,:]))
         
        for k in range(0,12):
            new_row.append(np.mean(chroma_cens[k,:]))
        
        new_row.append(np.mean(spec_cent))
        
        new_row.append(np.mean(spec_bw))
        
        for k in range(0,7):
            new_row.append(np.mean(spec_con[k,:]))

        new_row.append(np.mean(spec_flt))
    
        new_row.append(np.mean(rolloff))
        
        new_row.append(np.mean(zcr))

        for i in range(0,6):
            new_row.append(np.mean(tonnetz[i]))

        for e in mfcc:
            new_row.append(np.mean(e))

        new_row.append(b)

        tmp = pd.DataFrame(columns = l)
        
        for float_field in f:
            tmp[float_field] = tmp[float_field].astype(float, errors = 'raise')
        
        tmp.loc[len(tmp)] = new_row
        
        birds_fe = pd.concat([birds_fe, tmp], ignore_index = True, axis=0)  

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("\nEnd =", current_time)

print('\nFeatures extracted...')


# In[ ]:


# Dataframe features
birds_fe.drop(columns=['species', 'filename', 'duration'], inplace=True)
birds_fe.info()


# In[ ]:


# Encoding the Labels
genre_list = birds_fe.iloc[:, -1]
encoder = LabelEncoder()

# Fittng the data
y = encoder.fit_transform(genre_list)

# Dividing data into training and Testing set
X = birds_fe.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling the Feature columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Labels encoded. Train and Test Datasets, created and standardized.')


# In[ ]:


print('Random Forests Classifier\n')

now = datetime.now()
start = now.strftime("%H:%M:%S")
print("start =", start)

clf = RandomForestClassifier(bootstrap=False,
                             max_depth=21,
                             max_features='auto',
                             min_samples_leaf=1,
                             min_samples_split=2,
                             n_estimators=1800,
                             random_state=0
                            )

rfc = clf.fit(X_train, y_train)
      
# predicting labels
y_pred = clf.predict(X_test)

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

now = datetime.now()
finish = now.strftime("%H:%M:%S")
print("end =", finish)

print('Ready')


# In[ ]:


# Part 1:

# Directory where sound files have been placed
# test_audio_dir = '/kaggle/input/birdclef-2022/test_soundscapes/'
test_audio_dir = '../input/birdclef-2022/test_soundscapes/'

# All sound files will be splitted into 5-second chunks
chunk_size = 5   

# Getting all the file names from directory
file_list = [f.split('.')[0] for f in sorted(os.listdir(test_audio_dir))]
print('Number of test soundscapes found:', len(file_list))

# extracted features names
l = ['rmse' ]

for f in ['chroma_stft', 'chroma_cqt', 'chroma_cens']:
    for i in range(0, 12):
        l += [f+str(i)]
l += ['spec_cent', 'spec_bw']

for i in range(0,7):
        l += ['spec_con' + str(i)]

l += ['spec_flt', 'rolloff', 'zcr']

for i in range(0, 6):
        l += ['tonnetz' + str(i)]

for i in range(0, 20):
        l += ['mfcc' + str(i)]

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
        print("Chunk Nº ", counter, " - Chunk End Time ", chunk_end_time)

        # chunk_features holds all extracted features from the chunk.
        # this file is fully rewritten for each chunk
        
        chunk_features = []

        # Feature extraction from chunk
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=chunk, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=chunk, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
        spec_con = librosa.feature.spectral_contrast(y=chunk, sr=sr)
        spec_flt = librosa.feature.spectral_flatness(y=chunk)
        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=chunk)
        tonnetz = librosa.feature.tonnetz(y=chunk, sr=sr)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr)
        
        # Extract chunk features
        
        chunk_features = [np.mean(rms)]
        
        for k in range(0,12):
            chunk_features.append(np.mean(chroma_stft[k,:]))
        
        for k in range(0,12):
            chunk_features.append(np.mean(chroma_cqt[k,:]))
         
        for k in range(0,12):
            chunk_features.append(np.mean(chroma_cens[k,:]))
        
        chunk_features.append(np.mean(spec_cent))
        
        chunk_features.append(np.mean(spec_bw))

        for k in range(0,7):
            chunk_features.append(np.mean(spec_con[k,:]))

        chunk_features.append(np.mean(spec_flt))
    
        chunk_features.append(np.mean(rolloff))
        
        chunk_features.append(np.mean(zcr))

        for i in range(0,6):
            chunk_features.append(np.mean(tonnetz[i]))

        for e in mfcc:
            chunk_features.append(np.mean(e))

        # Scaling the features extracted from the chunk
        chunk_scaled = scaler.transform([chunk_features])
        
        # and predicting labels
        chunk_pred = clf.predict_proba(chunk_scaled) # Random Forest Model
        
        max_val = chunk_pred[0][np.argmax(chunk_pred)]
        j = 0
        for k in clf.classes_:
            bird = birds[k]
            row_id = afile + '_' + bird + '_' + str(chunk_end_time)
            
            # Put the result into our prediction dict
            dict_pred['row_id'].append(row_id)
            dict_pred['target'].append(True if chunk_pred[0][j]>=max_val else False)
            j += 1

        # next chunk
        counter += 1
        samples_wrote += buffer

# Part 3

# All sound files have been now splitted and the chunks, predicted.
# With the resulting dictionary make a new data frame and look at some results

results = pd.DataFrame(dict_pred, columns = ['row_id', 'target'])

# Convert results to csv
results.to_csv("/kaggle/working/submission.csv", index=False)

print('\nProcess ended successfully')
