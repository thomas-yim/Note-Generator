"""
Thomas Yim
4/14/2020
This file generates a pickle with randomized and processed data. It then trains
the model for the keyboard

"""

import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from cfg import Config
from createDf import createDataFrameFromJson
import librosa


"""
This checks if there is a pickle so that it does not process the data every
time the program runs. This is to save an hour on every run.
It returns the pickle if it exists or none.
"""
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

"""
This function is modeled after the code from Deep Learning for Audio Classification
On hundreds of audio of audio files, it can take an hour to process all the data
It randomizes all the data and creates a pickle to run program without the hour wait
"""
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.pitches==rand_class].index)
        rate, wav = wavfile.read(instrument + '_clean/'+file)
        label = df.at[file, 'pitches']
        if wav.shape[0]-config.step > 0:
            rand_index = np.random.randint(0, wav.shape[0]-config.step)
            sample = wav[rand_index:rand_index+config.step]
            # edited to use librosa.cqt instead of mfcc
            X_sample = librosa.cqt(sample, sr=rate)
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=88)
    config.data = (X,y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X,y

"""
This returns an instance of a model that has not been trained yet.
"""
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), 
                     padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), 
                     padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

#Ask the user for the instrument.
instrument = input("Input an instrument here: ")
#This examples.json file shows what files are labeled with what note
jsonPath = "nsynth-valid.jsonwav/nsynth-valid/examples.json"
#All wav files for training are in instrument_clean directories.
audioDir = instrument + "_clean/"
#This function takes all audio files and their label and puts it in one pandas df
df = createDataFrameFromJson(audioDir, jsonPath, instrument)

#This generates a list of [21, 22, 23, 24,..., 109]
classes = list(np.unique(df.pitches))
#This is to see how much data we have for each of the notes
class_dist = df.groupby(['pitches'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

"""
Note: We use the config so that when we test the accuracy in predict.py, we know
what the min and max values are. This is necessary to scale the values from
0 to 1.
"""
config = Config(mode=instrument)
X, y = build_rand_feat()
"""
Each row in the y array is an array that shows what note it is labeled with.
Structured like this:
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
It finds the index of the 1.
"""
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2],1)
model = get_conv_model()
    
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

'''
If you add this code back in you need to add it back to model.fit
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1,
                             mode='max', save_best_only=True,
                             save_weights_only=False, period=1)
'''

model.fit(X, y, epochs=20, shuffle=True, verbose=1)

#This saves the model to be used later in predict and songTranslator
model.save(config.model_path)
