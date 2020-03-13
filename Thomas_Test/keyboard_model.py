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
from tensorflow.keras.callbacks import ModelCheckpoint
from cfg import Config
from createDf import createDataFrameFromJson

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

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
        rate, wav = wavfile.read('keyboard_electronic_clean/'+file)
        label = df.at[file, 'pitches']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=88)
    config.data = (X,y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X,y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
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

jsonPath = "nsynth-valid.jsonwav/nsynth-valid/examples.json"
audioDir = "keyboard_electronic_clean/"
df = createDataFrameFromJson(audioDir, jsonPath)

classes = list(np.unique(df.pitches))
class_dist = df.groupby(['pitches'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='conv')
X, y = build_rand_feat()
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2], 1)
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

model.fit(X, y, epochs=8, shuffle=True, verbose=2, validation_split=0.1,)

model.save(config.model_path)
