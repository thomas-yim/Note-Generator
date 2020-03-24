import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from cfg import Config
import librosa
from createDf import createDataFrameFromJson

# checks to see if a pickles file exists that holds the data to be fed into the model
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

# builds the data that will be fed into the model
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.pitch==rand_class].index)
        rate, wav = wavfile.read('valid_clean_guitar/'+file)
        label = df.at[file, 'pitch']
        if wav.shape[0]-config.step > 0:
            rand_index = np.random.randint(0, wav.shape[0]-config.step)
            sample = wav[rand_index:rand_index+config.step]
            # edited to use librosa.cqt instead of mfcc
            X_sample = abs(librosa.cqt(sample, sr=rate))
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
    config.data = (X, y)
    
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    return X, y

# creates the machine learning model
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), 
                     padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), 
                     padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), 
                     padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(88, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

config = Config()

df = createDataFrameFromJson('valid_clean_guitar/', 'valid_labels.json')

classes = list(np.unique(df.pitches))
class_dist = df.groupby(['pitches'])['length'].mean()
n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()

# building the data set and model
X, y = build_rand_feat()
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2], 1)
model = get_conv_model()

class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

# setting up the model saving proecess
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, 
                             mode='max', save_best_only=True,
                             save_weights_only=False, period=1)

# running the model
model.fit(X, y, epochs=10, batch_size=32,
          shuffle=True, class_weight=class_weight,
          validation_split=0.1, callbacks=[checkpoint])

# saving the model
model.save(config.model_path)