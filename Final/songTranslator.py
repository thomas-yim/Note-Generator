import noteType as nt
import splitAudio as sa
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm

def get_notes(df, model_path, signal, sr):
    model = load_model(model_path)
    step = int(sr/10)
    pitch = []
    for i in tqdm(df.index):
        if df['pitch'][i] == 0:
            pitch.append(0)
        else:
            cqt = []
            _min, _max = float('inf'), -float('inf')
            for t in range(df['start'][i], df['end'][i] - step, step):
                sample = signal[t:t+step]
                x = np.abs(librosa.cqt(sample, sr=sr))
                _min = min(np.amin(x), _min)
                _max = max(np.amax(x), _max)
                cqt.append(x)
            cqt = (np.array(cqt) - _min)/(_max - _min)
            cqt = cqt.reshape(cqt.shape[0], cqt.shape[1], cqt.shape[2], 1)
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
            
def recognizeSong(filename):
    song_path = 'testsongs/' + filename + '.wav'
    model_path = 'models/guitar.model'
        
    signal, rate = librosa.load(song_path, 16000)
    
    df = pd.DataFrame(sa.split_into_chunks(signal, rate))
    
    note_mask = []
    for p in df['pitch']:
        if p == 1:
            note_mask.append(True)
        else:
            note_mask.append(False)
    bpm = nt.get_bpm(np.array(df['start'][note_mask]), rate)
    
    
    df.insert(1, 'length', nt.classify_note_types(np.array(df['start']), np.array(df['end']), rate, bpm))
    
    df['pitch'] = get_notes(df, model_path, signal, rate)
    
    print()
    print()
    print("Estimated tempo: " + str(bpm))
    return df