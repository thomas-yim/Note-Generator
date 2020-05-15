import os
import librosa
import json
import pandas as pd
from scipy.io import wavfile

import clean

inst_filt = str(input("Type instrument you want to view: "))

with open ('nsynth-test/examples.json', 'r') as f:
    notes = json.load(f)

fnames = []
pitches = []
for file in notes:
    if file[:len(inst_filt)] == inst_filt:
        fnames.append(file+'.wav')
        pitches.append(notes[file]['pitch'])

dataframe = {'fname': fnames, 'pitch': pitches}
df = pd.DataFrame(dataframe, columns=['fname', 'pitch'])

clean.cleanData(df, 'nsynth-test/audio', inst_filt + '_clean')