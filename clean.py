import os
import librosa
def cleanData(df, fromFolder, toFolder, os):
    if len(os.listdir("clean")) == 0:
        for f in tqdm(df.fname):
            signal, rate = librosa.load(fromFolder + '/' + f, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename=toFolder + "/" + f, rate=rate, data=signal[mask])