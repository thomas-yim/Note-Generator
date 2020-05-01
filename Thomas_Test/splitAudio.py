import pandas as pd
import noteType
import numpy as np
def split_into_chunks(signal, sr):
    
    #CONSTANTS
    #Check for the maximum ever 1/64 of a second
    rollingSize = int(sr/64)
    
    #The lowest max value for a rolling window that classifies as silence
    silenceThresh = 0.1
    
    #Gets rid of all the negatives because it is a mirror image across x-axis
    signal = list(abs(signal))
    
    """
    The following code will use a rolling window and take the max to normalize the data
    There will only 1/rollingSize left of the data, but it is okay to leave out that
    many points because there are 16000 points per second to use
    """
    k=0
    maxValues=[]
    #Since the area is rolling, we need to stop it before it goes over
    while k < len(signal)-rollingSize:
        #Find the maximum value to see where it spikes
        _max = max(signal[k:k+rollingSize])
        maxValues.append(_max)
        #Roll onto the next 100
        k += rollingSize
    
    sixteenth = int(sr/16)
    
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
    
    for j in range(0, len(spikeLocations)-1):
        change = spikeLocations[j+1] - spikeLocations[j]    
        #If the time between the supposed start of two notes is less than a set
        #time, then it wont be added to the new array of note starts
        if change > sixteenth:
            #This is to find the length of the shortest note to compare others to
            if change < _min:
                _min = change
            noteStarts.append(spikeLocations[j+1])
            noteTypes.append(1)
    
            
    """
    The following code checks for any rests
    
    endings = []
    notesAndRestStarts = []
    for i in range(0, len(noteStarts)-1):
        notesAndRestStarts.append(noteStarts[i])
        noteTypes.append(1)
        for maxIndex in range(noteStarts[i], noteStarts[i+1], rollingSize):     
            if (maxValues[int(maxIndex/rollingSize)+1] < silenceThresh):
                endings.append(maxIndex+1)
                notesAndRestStarts.append(maxIndex+1)
                endings.append(noteStarts[i+1])
                noteTypes.append(0)
                break
        
    notesAndRestStarts.append(noteStarts[i+1])
    noteTypes.append(1)
    """
    endings = noteStarts[1:]
    """
    The previous loops get the starts of all notes and the ends of all but the last note
    This just checks to see when it gets quiet enough where the note should end.
    """
    if len(signal) - noteStarts[-1] < _min:
        finalEnd = len(signal)
    else:
        finalEnd = noteStarts[-1]
        while len(signal) - finalEnd > _min:
            if max(signal[finalEnd : finalEnd + _min]) < silenceThresh:
                break
            finalEnd += _min
    
    endings.append(finalEnd)
    #Put it in a dataframe to be used by noteType.py
    dataframe = {'pitch':noteTypes, 'start':noteStarts, 'end':endings}
    df = pd.DataFrame(dataframe, columns=['pitch', 'start', 'end'])
    
    note_mask = []
    for p in df['pitch']:
        if p == 1:
            note_mask.append(True)
        else:
            note_mask.append(False)
    bpm = noteType.get_bpm(np.array(df['start'][note_mask]), sr)
    print(sr/bpm)
    tempoStarts = [df['start'][0]]
    for i in range(1, len(df['start'])):
        tempoStarts.append(tempoStarts[i-1] + (sr*60/bpm))
    df.insert(2, "Tempo Starts", tempoStarts)
    return df