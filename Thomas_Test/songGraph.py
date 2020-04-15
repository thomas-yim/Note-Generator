import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
from librosa import display
import json
import clean

def plot_signals(signal):
    plt.subplots(figsize=(20,5))
    plt.plot(signal)

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

def graphCQT(signal, rate):
    C = np.abs(librosa.cqt(signal, sr=rate))
    plt.subplots(figsize=(20,5))
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()
    return C

def plot_fft(fft):
    plt.subplots(figsize=(20,5))
    data = list(fft.values())[0]
    Y, freq = data[0], data[1]
    plt.plot(freq, Y)
            
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def printSplitGraphs(df):
    for i in range(0, len(df)):
        row = df.iloc[i]
        plot_signals(listSignal[row['start']:row['end']])
        plt.show()

signals = {}
fft = {}
signal, rate = librosa.load('songfiles/test3.wav', sr=16000)

mask = envelope(signal, rate, 0.0005)
signal = signal[mask]
signals["test"] = signal

plot_signals(list(signal))
plt.show()

C = graphCQT(list(signals.values())[0], 16000)
#Gets rid of all the negatives because it is a mirror image across x-axis
signal[signal < 0] = 0.0

fft["test"] = calc_fft(signal, rate)

#Index of the last nonzero value
last = 0
listSignal = list(signal)
for i in range(0,len(signal)):
    if listSignal[i] != 0.0:
        #Find the slope between the two points
        slope = (listSignal[i]-listSignal[last])/(i-last)
        for j in range(last, i-1):
            listSignal[j+1] = listSignal[j] + slope
        last = i

plot_signals(listSignal)
plt.show()

k=0
maxValues=[]
#Check for the maximum ever 1/64 of a second
rollingSize = int(16000/64)
#Since the size of the rolling window is 100, we subtract 100 from the length
while k < len(listSignal)-rollingSize:
    #Find the maximum value to see where it spikes
    _max = max(listSignal[k:k+rollingSize])
    maxValues.append(_max)
    #Roll onto the next 100
    k += rollingSize


plot_signals(maxValues)
plt.show()

sixteenth = 16000/16

valueChanges = []
for ii in range(0, len(maxValues)-1):
    if maxValues[ii+1] - maxValues[ii] > max(maxValues)/4:
        valueChanges.append(ii*rollingSize)

cleanList = []
noteTypes=[]
cleanList.append(valueChanges[0])
noteTypes.append(1)
_min = float('inf')
for j in range(0, len(valueChanges)-1):
    change = valueChanges[j+1] - valueChanges[j]    
    if change > sixteenth:
        if change < _min:
            _min = change
        cleanList.append(valueChanges[j+1])
        noteTypes.append(1)
print(_min)
for l in range(0, len(cleanList)-1):
    if (cleanList[l+1] - cleanList[l])/_min > 1.5:
        if max(listSignal[cleanList[l] + _min : cleanList[l+1]]) < 0.1:
            cleanList.insert(l+1, cleanList[l] + _min)
            noteTypes.insert(l+1, 0)



if len(listSignal) - cleanList[-1] < _min:
    finalEnd = len(listSignal)
else:
    finalEnd = cleanList[-1]
    while len(listSignal) - finalEnd > _min:
        if max(listSignal[finalEnd : finalEnd + _min]) < 0.1:
            break
        finalEnd += _min

endings = cleanList[1:]
endings.append(finalEnd)

dataframe = {'type':noteTypes, 'start':cleanList, 'end':endings}
df = pd.DataFrame(dataframe, columns=['type', 'start', 'end'])

printSplitGraphs(df)

'''
NOTE: THE FOLLOWING CODE IS A WORKING VERSION THAT IS LESS VERSATILE
THRESHOLDING MIN VERSUS MAX WITHIN A ROLLING WINDOW OF 100
changes = []
k = 0
With the line "if (_max- _min > 0.3)", the 0.3 is the threshold. This will work
if we have the right value for that one.
#Since the size of the rolling window is 100, we subtract 100 from the length
while k < len(listSignal)-100:
    #Find the maximum and minimum value to see where it spikes
    _max = max(listSignal[k:k+100])
    _min = min(listSignal[k:k+100])
    #If the difference in the value is more than a third of the max, consider it a spike
    if (_max - _min > 0.3):
        changes.append(k)
    #Roll onto the next 100
    k += 100
cleanList = []
cleanList.append(changes[0])
print(max(changes))
for i in range(0,len(changes)-1):
    #print(changes[i+1]-changes[i])
        cleanList.append(changes[i+1])
'''
'''     
GRAPHS OF THE INDIVIDUAL ROWS OF A CONSTANT Q TRANSFORM
plt.subplots(figsize=(20,5))
plt.plot(C[0])
plt.subplots(figsize=(20,5))
plt.plot(C[40])
plt.subplots(figsize=(20,5))
plt.plot(C[81])
'''

'''
SEE IF AVERAGE DECIBLE VALUE FROM CQT CHANGES OBVIOUSLY
This is code to find what the average decible is along the vertical spectrum to
see where the change might appear (Note: there is no visible pattern here)
for i in range(0, len(C[81]-1)):
    totalSum = 0
    for j in range(0, len(C)-1):
        totalSum += C[j][i]
    print(abs(totalSum/84))
'''

'''
LOOK FOR SPIKES IN DECIBLES FROM THE CQT GRAPH
This is code to check for notes using a minimum threshold change in decibles.
(Note: this did not eventually work)
count = 0
for i in range(0,len(C[81])-1):
    if i != 0:
        if abs((C[81][i]-C[81][i-1]) > 10):
            count += 1
            
print(count)
'''
    
    