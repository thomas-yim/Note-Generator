import os
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean < threshold:
            mask.append(False)
        else:
            mask.append(True)
    return mask

def cleanData(df, fromFolder, toFolder):
    if len(os.listdir(toFolder)) == 0:
        for f in tqdm(df.fname):
            signal, rate = librosa.load(fromFolder + '/' + f, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename=toFolder + "/" + f, rate=rate, data=signal[mask])