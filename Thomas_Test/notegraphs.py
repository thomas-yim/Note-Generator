import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import json
import clean
from createDf import createDataFrameFromJson

"""
The following four functions are ways to visualize the audio.
plot_signals prints the signal with the x-axis as time
plot_fft prints the fast fourier transform. It is basically a function
    that takes the audio and creates a unique audio fingerprint. But
    we never ended up using this because the constant q transform actually
    does a better job of separating notes (cqt has a 84,4 structure while fft
    only has 13,8 and more data means more features for the model)
plot_mfcc and plot_fbank help with the fft
Note: These were taken from the audio classification tutorial on YouTube
"""
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=10, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(10):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)    
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

#This adds a "_clean" or "_clean_test" to the end to be later used by the model
#or prediction program.
trainData = input("Is this for a train dataset (True or False case sensitive): ")
print(type(trainData))
if (trainData == "True"):
    jsonPath = "nsynth-valid.jsonwav/nsynth-valid/examples.json"
    audioDir = "nsynth-valid.jsonwav/nsynth-valid/"
    dataType = "_clean/"
else:
    jsonPath = "notefiles/examples.json"
    audioDir = "notefiles/"
    dataType = "_clean_test/"
instrument = input("Input an instrument here: ")
#directory = "nsynth-valid.jsonwav/nsynth-valid/"
df = createDataFrameFromJson(audioDir + "audio/", jsonPath, instrument)
    
classes = list(np.unique(df['pitches']))
class_dist = df.groupby(['pitches'])['length'].mean()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs =  {}

for c in classes:
    wav_file = df[df['pitches']==c].iloc[0,0]
    signal, rate = librosa.load(audioDir + 'audio/'+wav_file, sr=16000)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=400).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=400).T
    mfccs[c] = mel
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

print(audioDir)
#This populates a new directory with cleaned data. Cleaned data gets rid of extended periods of silence.
clean.cleanData(df, audioDir, instrument + dataType)
    
