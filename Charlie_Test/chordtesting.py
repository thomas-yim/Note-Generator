import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def graph_cqt(signal, rate):
    C = np.abs(librosa.cqt(signal, sr=rate))
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()
    cqts = separate_freqs(C)
    for cqt in cqts.values():
        librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                                 sr=rate, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.tight_layout()
        plt.show()
    return cqts

def separate_freqs(cqt):
    cqt = cqt.T
    maxes = np.zeros(84)
    for t in range(int(len(cqt)/2)):
        _max = cqt[t].max()
        for freq in range(len(cqt[t])):
            if cqt[t][freq] > 0.8*_max:
                maxes[freq] += 1
    print(maxes)
    peaks = []
    for i in range(len(maxes)):
        if maxes[i] > len(cqt)/4:
            peaks.append(i)
    if len(peaks) > 1:
        sep_cqts = {}
        for freq in peaks:
            for t_slice in cqt:
                flat_cqt = flatten_cqt(t_slice, peaks, freq)
                if freq in sep_cqts.keys():
                    sep_cqts[freq].append(flat_cqt)
                else:
                    sep_cqts.update({freq:[flat_cqt]})
        for freq in sep_cqts.keys():
            sep_cqts[freq] = np.array(sep_cqts[freq]).T
        return sep_cqts
    else:
        return {0: cqt.T}

def find_start(cqt, peaks, freq):
    if freq < 0:
        return -1
    elif (freq-1) in peaks:
        return find_start(cqt, peaks, freq-3)
    elif freq in peaks:
        return find_start(cqt, peaks, freq-2)
    elif (freq+1) in peaks:
        return find_start(cqt, peaks, freq-1)
    else:
        return freq

def find_end(cqt, peaks, freq):
    if freq > 83:
        return 84
    elif (freq-1) in peaks:
        return find_end(cqt, peaks, freq+1)
    elif freq in peaks:
        return find_end(cqt, peaks, freq+2)
    elif (freq+1) in peaks:
        return find_end(cqt, peaks, freq+3)
    else:
        return freq

def flatten_cqt(cqt, peaks, exclude_peak):
    new_cqt = []
    for f in cqt:
        new_cqt.append(f)
    new_cqt = np.array(new_cqt)
    while len(peaks) > 0:
        start_flat = find_start(cqt, peaks, peaks[0])
        end_flat = find_end(cqt, peaks, peaks[0])
        if start_flat == -1:
            for i in range(end_flat):
                if (i+1)!=exclude_peak and i!=exclude_peak and (i-1)!=exclude_peak:
                    new_cqt[i] = cqt[end_flat]
        elif end_flat == 84:
            for i in range(start_flat+1, end_flat):
                if (i+1)!=exclude_peak and i!=exclude_peak and (i-1)!=exclude_peak:
                    new_cqt[i] = cqt[start_flat]
        else:
            diff = (cqt[end_flat] - cqt[start_flat])/(end_flat-start_flat)
            for i in range(start_flat+1, end_flat):
                if (i+1)!=exclude_peak and i!=exclude_peak and (i-1)!=exclude_peak:
                    new_cqt[i] = cqt[start_flat] + (i-start_flat)*diff
        to_delete = []
        for p in range(len(peaks)):
            if peaks[p] > start_flat and peaks[p] < end_flat:
                to_delete.append(p)
        d = len(to_delete)-1
        while d >= 0:
            peaks = np.delete(peaks, to_delete[d])
            d -= 1
    return new_cqt

def normal_graph_cqt(signal, rate):
    C = np.abs(librosa.cqt(signal, sr=rate))
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()

def get_notes(model_path, cqt):
    model = load_model(model_path)
    _min = np.min(cqt)
    _max = np.max(cqt)
    cqt = (np.array(cqt) - _min)/(_max - _min)
    split_cqt = []
    t = 4
    while (t+3) < len((2/3)*cqt[0]):
        split_cqt.append((cqt.T[t:t+4]).T)
        t += 4
    split_cqt = np.array(split_cqt)
    split_cqt = split_cqt.reshape(split_cqt.shape[0], split_cqt.shape[1], split_cqt.shape[2], 1)
    likelihood = np.zeros(88)
    predict_pitches = model.predict(split_cqt)
    s = 0
    for segment in predict_pitches:
        for note in range(len(segment)):
            if len(predict_pitches) == 1 or s > 0:
                likelihood[note] += predict_pitches[s][note]
        s += 1
    return np.argmax(likelihood)

model_path = 'models/keyboard_electronic.model'

signal, rate = librosa.load('chords/chords2.wav')
cqts = graph_cqt(signal,rate)
for c in cqts.values():
    print(get_notes(model_path, c))

signal, rate = librosa.load('chords/ctest.wav')
normal_graph_cqt(signal,rate)
signal, rate = librosa.load('chords/etest.wav')
normal_graph_cqt(signal,rate)
