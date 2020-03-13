import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from createDf import createDataFrameFromJson
from tensorflow.keras.utils import to_categorical

def build_predictions(audio_dir):
    test_labels = []
    test_audio = []
    
    print("Extracting features from audio")
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x=mfcc(sample, rate, numcep=config.nfeat,
                   nfilt=config.nfilt, nfft=config.nfft).T
            x = (x - config.min) / (config.max - config.min)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            test_audio.append(x)
            test_labels.append(c)
    test_labels = to_categorical(test_labels, num_classes=88)
    return test_labels, test_audio

audio_dir = "keyboard_electronic_clean_test/"
jsonPath = "notefiles/examples.json"
df = createDataFrameFromJson(audio_dir, jsonPath)
classes = list(np.unique(df.pitches))
fn2class = dict(zip(df.index, df.pitches))
p_path = os.path.join("pickles", "conv.p")

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model=load_model(config.model_path)

test_labels, test_audio = build_predictions('keyboard_electronic_clean_test')
print(np.array(test_audio).shape)
score, acc = model.evaluate(np.array(test_audio), test_labels)

print(acc)
