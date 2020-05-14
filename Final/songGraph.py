import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display
from splitAudio import split_into_chunks, rollingMax
import matplotlib.pyplot as plt

#This plots the wav file so we can see where the spikes are
def plot_signals(signal):
    plt.subplots(figsize=(20,5))
    plt.plot(signal)


#This prints the graphs split at each note/rest
def printSplitGraphs(df, signal):
    for i in range(0, len(df)):
        row = df.iloc[i]
        cutSignal = signal[int(row['start']):int(row['end'])]
        plot_signals(cutSignal)
        plt.show()
        #Get the array of the rolling maximums from this section
        maxValues = rollingMax(cutSignal, int(16000/64))
        #Print the average rolling maximum from this section
        print(sum(maxValues)/len(maxValues))

signal, rate = librosa.load('testsongs/Guitar1.wav', sr=16000)


#This plots the normal signal. It is hard to see, but it crosses the x-axis frequently
plot_signals(list(signal))
plt.show()

#This is just to plot the upper half of signal
plot_signals(list(abs(signal)))


df = split_into_chunks(signal, rate)

#Print the graphs to see if we split it correctly
#printSplitGraphs(df, signal)
