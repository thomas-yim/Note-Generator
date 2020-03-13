import pandas as pd
import json
from scipy.io import wavfile

def createDataFrameFromJson(audio_dir, jsonPath):
    with open(jsonPath, 'r') as f:
        exampleNotes = json.load(f)
    filenames = []
    pitches = []
    instrument = input("Input an instrument here: ")
    for filename in exampleNotes:
        if (filename[:len(instrument)] == instrument):
            filenames.append(filename + ".wav")
            pitches.append(exampleNotes[filename]['pitch'])
    
    dataframe = {'fname':filenames, 'pitches': pitches}
    df = pd.DataFrame(dataframe, columns=['fname', 'pitches'])
    
    df.set_index('fname', inplace=True)
    
    for f in df.index:
        rate, signal = wavfile.read(audio_dir + f)
        df.at[f, 'length'] = signal.shape[0]/rate
    return df
    