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

#This plots the wav file so we can see where the spikes are
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

#This prints the graphs split at each note/rest
def printSplitGraphs(df):
    for i in range(0, len(df)):
        row = df.iloc[i]
        plot_signals(listSignal[row['start']:row['end']])
        plt.show()

signals = {}
signal, rate = librosa.load('songfiles/test4.wav', sr=16000)

mask = envelope(signal, rate, 0.0005)
signal = signal[mask]
signals["test"] = signal

#This plots the normal signal. It is hard to see, but it crosses the x-axis frequently
plot_signals(list(signal))
plt.show()


#Gets rid of all the negatives because it is a mirror image across x-axis
signal[signal < 0] = 0.0

#This will show the same graph, but it will be cut to show positive values
plot_signals(listSignal)
plt.show()

"""
The following code will use a rolling window and take the max to normalize the data
There will only 1/rollingSize left of the data, but it is okay to leave out that
many points because there are 16000 points per second to use
"""
k=0
maxValues=[]
#Check for the maximum ever 1/64 of a second
rollingSize = int(16000/64)
#Since the area is rolling, we need to stop it before it goes over
while k < len(listSignal)-rollingSize:
    #Find the maximum value to see where it spikes
    _max = max(listSignal[k:k+rollingSize])
    maxValues.append(_max)
    #Roll onto the next 100
    k += rollingSize

#This will show the result of the 
plot_signals(maxValues)
plt.show()

sixteenth = 16000/16

#This will contain all indices of places where the sound spikes
valueChanges = []
for i in range(0, len(maxValues)-1):
    #If the difference between two values is greater than a forth of the biggest one
    if maxValues[i+1] - maxValues[i] > max(maxValues)/4:
        #We multiply by rolling size to get the real index back from it
        valueChanges.append(i*rollingSize)

#This list will contain the indices of all the starts of the arrays
cleanList = []
#For each value in clean List, noteTypes will have a 1 (note) or a 0 (rest)
noteTypes=[]
#We know that the first spike is a note so we add that one
cleanList.append(valueChanges[0])
noteTypes.append(1)

#This will become equal to the smallest interval betwen notes (shortest note)
_min = float('inf')

"""
Note: The following code is necessary because the sigal spikes multiple times
at the start of each new note. Thus, the previous loop counts all of them. The
following code gets rid of the duplicates
"""
for j in range(0, len(valueChanges)-1):
    change = valueChanges[j+1] - valueChanges[j]    
    #If the time between the supposed start of two notes is less than a set
    #time, then it wont be added to the new array of note starts
    if change > sixteenth:
        #This is to find the length of the shortest note to compare others to
        if change < _min:
            _min = change
        cleanList.append(valueChanges[j+1])
        noteTypes.append(1)
        
"""
The following code checks for any rests
"""
for l in range(0, len(cleanList)-1):
    #If the time between two notes is longer than the shortest length by a large
    #amount, it may be a rest
    if (cleanList[l+1] - cleanList[l])/_min > 1.5:
        #For the time after a set note, if it drops off to less than 0.1, its a rest
        if max(listSignal[cleanList[l] + _min : cleanList[l+1]]) < 0.1:
            cleanList.insert(l+1, cleanList[l] + _min)
            noteTypes.insert(l+1, 0)

"""
The previous loops get the starts of all notes and the ends of all but the last note
This just checks to see when it gets quiet enough where the note should end.
"""
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

#Put it in a dataframe to be used by noteType.py
dataframe = {'type':noteTypes, 'start':cleanList, 'end':endings}
df = pd.DataFrame(dataframe, columns=['type', 'start', 'end'])

printSplitGraphs(df)

'''
NOTE: THE FOLLOWING CODE IS A WORKING VERSION THAT IS LESS VERSATILE...

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
    
    