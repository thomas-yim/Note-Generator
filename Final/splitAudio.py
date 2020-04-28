import pandas as pd

def split_into_chunks(signal, sr):
    
    #Gets rid of all the negatives because it is a mirror image across x-axis
    listSignal = list(abs(signal))
    
    """
    The following code will use a rolling window and take the max to normalize the data
    There will only 1/rollingSize left of the data, but it is okay to leave out that
    many points because there are 16000 points per second to use
    """
    k=0
    maxValues=[]
    #Check for the maximum ever 1/64 of a second
    rollingSize = int(sr/64)
    #Since the area is rolling, we need to stop it before it goes over
    while k < len(listSignal)-rollingSize:
        #Find the maximum value to see where it spikes
        _max = max(listSignal[k:k+rollingSize])
        maxValues.append(_max)
        #Roll onto the next 100
        k += rollingSize
    
    sixteenth = int(sr/16)
    
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
    
    #This will become equal to the smallest interval betwen notes (shortest note)
    _min = float('inf')
    
    """
    Note: The following code is necessary because the signal spikes multiple times
    at the start of each new note. Thus, the previous loop counts all of them. The
    following code gets rid of the duplicates
    """
    #This list will contain the indices of all the starts of the arrays
    cleanList = []
    #For each value in clean List, noteTypes will have a 1 (note) or a 0 (rest)
    noteTypes=[]
    #We know that the first spike is a note so we add that one
    cleanList.append(valueChanges[0])
    
    for j in range(0, len(valueChanges)-1):
        change = valueChanges[j+1] - valueChanges[j]    
        #If the time between the supposed start of two notes is less than a set
        #time, then it wont be added to the new array of note starts
        if change > sixteenth:
            #This is to find the length of the shortest note to compare others to
            if change < _min:
                _min = change
            cleanList.append(valueChanges[j+1])
    
            
    """
    The following code checks for any rests
    """
    endings = []
    notesAndRestStarts = []
    for l in range(0, len(cleanList)-1):
        notesAndRestStarts.append(cleanList[l])
        noteTypes.append(1)
        for maxIndex in range(cleanList[l], cleanList[l+1], rollingSize):     
            if (maxValues[int(maxIndex/rollingSize)+1] < 0.1):
                endings.append(maxIndex+1)
                notesAndRestStarts.append(maxIndex+1)
                endings.append(cleanList[l+1])
                noteTypes.append(0)
                break
        
    notesAndRestStarts.append(cleanList[l+1])
    noteTypes.append(1)
    
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
    
    endings.append(finalEnd)
    
    #Put it in a dataframe to be used by noteType.py
    dataframe = {'pitch':noteTypes, 'start':notesAndRestStarts, 'end':endings}
    df = pd.DataFrame(dataframe, columns=['pitch', 'start', 'end'])
    return df