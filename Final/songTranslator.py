
import noteType as nt
import splitAudio as sa
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm

#This takes each of the notes and runs the model on it
def get_notes(df, model_path, signal, sr):
    #Load the pre-existing model. instrument.model
    model = load_model(model_path)
    #The model recognizes with 1/10 of a second of the note
    step = int(sr/10)
    pitch = []
    print("model: ", model)
    #For each of the notes given by splitaudio.py
    for i in tqdm(df.index):
        #If it is a rest, append a 0
        if df['type'][i] == 0:
            pitch.append(0)
        else:
            cqt = []
            _min, _max = float('inf'), -float('inf')
            #Since the max bpm we recognize is 140. There will be at least 4/10 seconds
            #of data. We can run the model on individual 1/10 second steps and
            #Use those to increase the overall accuracy by comparing predicitons.
            for t in range(df['start'][i], df['end'][i] - step, step):
                sample = signal[t:t+step]
                x = np.abs(librosa.cqt(sample, sr=sr))
                _min = min(np.amin(x), _min)
                _max = max(np.amax(x), _max)
                cqt.append(x)
            #Must be scaled from 0 to 1
            cqt = (np.array(cqt) - _min)/(_max - _min)
            cqt = cqt.reshape(cqt.shape[0], cqt.shape[1], cqt.shape[2], 1)
            print(cqt.shape[0], cqt.shape[1], cqt.shape[2], 1)
            predict_pitches = model.predict(cqt)
            likelihood = np.zeros(88)
            s = 0
            for segment in predict_pitches:
                for note in range(len(segment)):
                    if len(predict_pitches) == 1 or s > 0:
                        likelihood[note] += predict_pitches[s][note]
                s += 1
            pitch.append(np.argmax(likelihood) + 21)
    return pitch
            
def recognizeSong(filename, instrument):
    song_path = 'testsongs/' + filename + '.wav'
    model_path = 'models/' + instrument + '.model'
        
    signal, rate = librosa.load(song_path, 16000)
    
    #Split it into its individual notes and rests (with correct lengths)
    df = pd.DataFrame(sa.split_into_chunks(signal, rate))
    print(df.head)
    note_mask = []
    for p in df['type']:
        if p == 1:
            note_mask.append(True)
        else:
            note_mask.append(False)
    bpm = nt.get_bpm(np.array(df['start'][note_mask]), rate)
    
    #This finds the length of a note (quarter, half, whole) from the bpm
    df.insert(1, 'length', nt.classify_note_types(np.array(df['start']), np.array(df['end']), rate, bpm))
    
    df['type'] = get_notes(df, model_path, signal, rate)
    print(df.head)
    print()
    print()
    print("Estimated tempo: " + str(bpm))
    return df