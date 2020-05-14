import pandas as pd
import noteType
import numpy as np
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
        print(maxValues[0])
        print(maxValues[int(len(maxValues)/2)])

"""
The following code will use a rolling window and take the max to normalize the data
There will only 1/rollingSize left of the data, but it is okay to leave out that
many points because there are 16000 points per second to use
"""
def rollingMax(signal, rollingSize):
    k=0
    maxValues=[]
    #Since the area is rolling, we need to stop it before it goes over
    while k < len(signal)-rollingSize:
        #Find the maximum value to see where it spikes
        _max = max(signal[k:k+rollingSize])
        maxValues.append(_max)
        #Roll onto the next 100
        k += rollingSize
    return maxValues

"""
This returns a dataframe with the audio split into its starts, ends, and note/not note.
"""
def split_into_chunks(signal, sr):
    
    #CONSTANTS
    #Check for the maximum every 1/64 of a second
    rollingSize = int(sr/64)
    
    signal = list(signal)
    
    maxValues = rollingMax(signal, rollingSize)
    
    #This will contain all indices of places where the sound spikes
    spikeLocations = []
    for i in range(0, len(maxValues)-1):
        #If the difference between two values is greater than a forth of the biggest one
        #Note this may not work for all audio files but seems to be good for garageband music
        if maxValues[i+1] - maxValues[i] > max(maxValues)/4:
            #We multiply by rolling size to get the real index back from it
            spikeLocations.append(i*rollingSize)
    
    
    """
    Note: The following code is necessary because the signal spikes multiple times
    at the start of each new note. Thus, the previous loop counts all of them. The
    following code gets rid of the duplicates
    """
    #This list will contain the indices of all the starts of the arrays
    noteStarts = []
    #For each value in clean List, noteTypes will have a 1 (note) or a 0 (rest)
    noteTypes=[]
    #We know that the first spike is a note so we add that one
    noteStarts.append(spikeLocations[0])
    noteTypes.append(1)
    
    #This will become equal to the smallest interval betwen notes (shortest note)
    _min = float('inf')
    
    #This will make sure that notes are at least 1/16 a second apart
    #This is fine because a note at 140 bpm is just over 2/5 of a second.
    sixteenth = int(sr/16)
    
    for j in range(0, len(spikeLocations)-1):
        change = spikeLocations[j+1] - spikeLocations[j]    
        #If the time between the supposed start of two notes is at least 1/16
        #of a second, then it will be added to the new array of note starts
        if change > sixteenth:
            #This is to find the length of the shortest note to later
            #find when the audio file ends
            if change < _min:
                _min = change
            noteStarts.append(spikeLocations[j+1])
            noteTypes.append(1)
    
            
    #A note ends when another starts
    endings = noteStarts[1:]
    """
    The previous loops get the starts of all notes and the ends of all but the last note
    This just checks to see when it gets quiet enough where the note should end.
    """
    #The lowest max value for a rolling window that classifies as silence
    silenceThresh = 0.02
    #make sure the file is at least one more note long
    if len(signal) - noteStarts[-1] < _min:
        finalEnd = len(signal)
    else:
        #If it reached here, there is at least another note of length left
        finalEnd = noteStarts[-1]
        while len(signal) - finalEnd > _min:
            #If it dies down to under our threshold, it is silence
            if max(signal[finalEnd : finalEnd + _min]) < silenceThresh:
                break
            finalEnd += _min
    
    endings.append(finalEnd)
    #Put it in a dataframe to be used by noteType.py
    dataframe = {'type':noteTypes, 'start':noteStarts, 'end':endings}
    df = pd.DataFrame(dataframe, columns=['type', 'start', 'end'])
    note_mask = []
    for p in df['type']:
        #p will equal 1 when it is a note not a rest
        if p == 1:
            note_mask.append(True)
        else:
            note_mask.append(False)
    #When there are a lot of notes, the transient probability in noteType
    #does not work as well so if it is over 10 notes, we take 5 from it.
    #The reason it is 3:8 is it takes the middle 5 of that 10.
    if len(df['start']) > 10:
        bpm = noteType.get_bpm(np.array(df['start'][note_mask][3:8]), sr)
    else:
        bpm = noteType.get_bpm(np.array(df['start'][note_mask]), sr)
    print("BPM: " + str(bpm))
    
    #This is the number of data points per beat. 16000*60/bpm
    tempo = int(sr*60/bpm)
    print("Tempo: " + str(tempo))
    #These will be the possible starts of notes according to the tempo
    starts = []
    #These are the ends according to the tempo
    endings = []
    noteTypes=[]
    #While the start is less than the last start plus an extra bar (a bar=4 quarters)
    #Calculate possible starts based on tempo
    for k in range(0, len(df['start'])):
        if k != len(df['start'])-1:
            #If the difference between two notes is longer than a quarter note, check for rests
            if int(round((df['start'][k+1] - df['start'][k])/tempo,0)) >= 2:
                checkForRestIndex = df['start'][k]
                #This becomes the rolling average. If it increases, it is always a new note
                previousAvg = 0
                #While there is still a quarter note of length unlabeled 
                while int(round((df['start'][k+1] - checkForRestIndex)/tempo,0)) > 0:
                    starts.append(checkForRestIndex)
                    if int(round((df['start'][k+1] - checkForRestIndex)/tempo,0)) == 1:
                        end = df['start'][k+1]
                    else:
                        end = checkForRestIndex + tempo
                    endings.append(end)
                    cutSignal = signal[checkForRestIndex:end]
                    maxValues = rollingMax(cutSignal, int(16000/64))
                    average = sum(maxValues)/len(maxValues)
                    #If the average increases, it is a new note
                    if average > previousAvg:
                        noteTypes.append(1)
                    #If the last note is a rest and it doesn't increase, the next is a rest
                    elif noteTypes[-1] == 0:
                        noteTypes.append(0)
                    #If the max at the start/max halfway through the sound < 2
                    #then it is a held note
                    elif maxValues[0]/maxValues[int(len(maxValues)/2)] < 2:
                        noteTypes.append(-1)
                    #else it is a rest
                    else:
                        noteTypes.append(0)
                    previousAvg = average
                    checkForRestIndex += tempo
            #If it wasn't longer than a quarter, add the start and end
            else:
                starts.append(df['start'][k])
                endings.append(df['start'][k+1])
                noteTypes.append(1)
        else:
            """
            Note this code is similar to the above, BUT it is necessary to handle
            the end of the file. The final boundary is not a new note because
            it is the last note.
            """
            checkForRestIndex = df['start'][k]
            previousAvg = 0
            #We check for one more bar of length because there won't be more than
            # a bar of rests at the end of a song.
            finalEnd = df['start'][k]+4*tempo
            print(finalEnd)
            while int(round((finalEnd - checkForRestIndex)/tempo,0)) > 0:
                starts.append(checkForRestIndex)
                if int(round((finalEnd - checkForRestIndex)/tempo,0)) == 1:
                    end = finalEnd
                else:
                    end = checkForRestIndex + tempo
                endings.append(end)
                cutSignal = signal[checkForRestIndex:end]
                maxValues = rollingMax(cutSignal, int(16000/64))
                average = sum(maxValues)/len(maxValues)
                if average > previousAvg:
                    noteTypes.append(1)
                elif noteTypes[-1] == 0:
                    noteTypes.append(0)
                elif maxValues[0]/maxValues[int(len(maxValues)/2)] < 2:
                    noteTypes.append(-1)
                else:
                    noteTypes.append(0)
                previousAvg = average
                checkForRestIndex += tempo
            
    
    #Add this to a new dataframe where everything is determined by tempo
    tempodf = pd.DataFrame({'start':starts,'end':endings}, columns=['start','end'])
    
    #Add this type column to the dataframe
    tempodf.insert(0, "type", noteTypes)
    endOfFile = tempodf.iloc[-1]['end']
    #This deletes all rows where it is a continuation of a held note
    tempodf = tempodf[tempodf.type != -1]
    #This is df['end'] = df['start'][1:] but they aren't hashable so we loop it
    for j in range(1, len(tempodf)):
        tempodf.iloc[j-1]['end'] = tempodf.iloc[j]['start']
    tempodf.iloc[-1]['end'] = endOfFile
    
    return tempodf