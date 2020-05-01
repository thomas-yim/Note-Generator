import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display
from splitAudio import split_into_chunks, rollingMax

#This plots the wav file so we can see where the spikes are
def plot_signals(signal):
    plt.subplots(figsize=(20,5))
    plt.plot(signal)


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

#This prints the graphs split at each note/rest
def printSplitGraphs(df, signal):
    for i in range(0, len(df)):
        row = df.iloc[i]
        cutSignal = signal[row['start']:row['end']]
        plot_signals(cutSignal)
        plt.show()
        maxValues = rollingMax(cutSignal, int(16000/64))
        print(sum(maxValues)/len(maxValues))

signal, rate = librosa.load('testsongs/n4r4r2n2r2n1.wav', sr=16000)


#This plots the normal signal. It is hard to see, but it crosses the x-axis frequently
plot_signals(list(signal))
plt.show()


plot_signals(list(abs(signal)))

df = split_into_chunks(signal, rate)
tempo = df.iloc[1]["Tempo Starts"] - df.iloc[0]["Tempo Starts"]
currentStart = df.iloc[0]['start']
starts = []
endings = []
while currentStart < df.iloc[len(df['start'])-1]['start']+5*tempo:
    starts.append(currentStart)
    endings.append(currentStart+tempo)
    currentStart += tempo

newdf = pd.DataFrame({'start':starts,'end':endings}, columns=['start','end'])
    
for i in range(0, len(df)):
    row = df.iloc[i]
    cutSignal = signal[row['start']:row['end']]
    maxValues = rollingMax(cutSignal, int(16000/64))
    average = sum(maxValues)/len(maxValues)
    
    
printSplitGraphs(newdf, signal)

'''
NOTE: THE FOLLOWING CODE IS a WORKING VERSIONS THAT IS LESS VERSATILE...

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
   
GRAPHS OF THE INDIVIDUAL ROWS OF A CONSTANT Q TRANSFORM
plt.subplots(figsize=(20,5))
plt.plot(C[0])
plt.subplots(figsize=(20,5))
plt.plot(C[40])
plt.subplots(figsize=(20,5))
plt.plot(C[81])

SEE IF AVERAGE DECIBLE VALUE FROM CQT CHANGES OBVIOUSLY
This is code to find what the average decible is along the vertical spectrum to
see where the change might appear (Note: there is no visible pattern here)
for i in range(0, len(C[81]-1)):
    totalSum = 0
    for j in range(0, len(C)-1):
        totalSum += C[j][i]
    print(abs(totalSum/84))

LOOK FOR SPIKES IN DECIBLES FROM THE CQT GRAPH
This is code to check for notes using a minimum threshold change in decibles.
(Note: this did not eventually work)
count = 0
for i in range(0,len(C[81])-1):
    if i != 0:
        if abs((C[81][i]-C[81][i-1]) > 10):
            count += 1
            
print(count)

THIS TRIED TO NORMALIZE THE DATA BY GETTING RID OF ALL THE SMALL VALUES
BETWEEN TWO LARGE ONES
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
'''
    
    