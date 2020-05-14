import pandas as pd
import json
from scipy.io import wavfile

"""
This is a function that takes the audio and the json with its info
and puts it into a dataframe with filename, pitch, and length columns.
The json file provided by the data base contains instrument type, pitch,
and many other data, but we only care about the pitch and filename
"""
def createDataFrameFromJson(audio_dir, jsonPath, instrument):
    with open(jsonPath, 'r') as f:
        exampleNotes = json.load(f)
    filenames = []
    pitches = []
    #This finds all the intruments from the given instrument
    for filename in exampleNotes:
        if (filename[:len(instrument)] == instrument):
            filenames.append(filename + ".wav")
            pitches.append(exampleNotes[filename]['pitch'])
    
    dataframe = {'fname':filenames, 'pitches': pitches}
    df = pd.DataFrame(dataframe, columns=['fname', 'pitches'])
    
    df.set_index('fname', inplace=True)
    
    #The rate is the number of data points per second
    for f in df.index:
        rate, signal = wavfile.read(audio_dir + f)
        df.at[f, 'length'] = signal.shape[0]/rate
    return df
    