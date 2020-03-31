import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from tensorflow.keras.models import load_model
import pandas as pd
from createDf import createDataFrameFromJson
from tensorflow.keras.utils import to_categorical
import librosa

class TestData:
    def __init__(self, instrument='keyboard_electronic'):
        self.instrument = instrument
        self.p_path = os.path.join('pickles', instrument + '_test.p')

def check_data():
    if os.path.isfile(testData.p_path):
        print('Loading existing data for {} test data'.format(testData.instrument))
        with open(testData.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp

def build_predictions(audio_dir):
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    test_labels = []
    test_audio = []
    
    print("Extracting features from audio")
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            # edited to use librosa.cqt instead of mfcc
            x = librosa.cqt(sample, sr=rate)
            x = (x - config.min) / (config.max - config.min)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            test_audio.append(x)
            test_labels.append(c)
    test_labels = to_categorical(test_labels, num_classes=len(classes))
    testData.data = (test_labels, test_audio)
    with open(testData.p_path, 'wb') as handle:
        pickle.dump(testData, handle, protocol=2)
    return test_labels, test_audio

instrument = input("Input an instrument here: ")
#Change this variable to wherever your cleaned test data is stored
audio_dir = instrument + "_clean_test"
#Change this variable to wherever the nsynth json file with labels is stored
jsonPath = "notefiles/examples.json"
#This is from another file. Puts the wav signal and label in a dataframe
df = createDataFrameFromJson(audio_dir + "/", jsonPath, instrument)
#Get a list of all the possible labels
classes = list([21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,
                44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,
                66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,
                88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108])
#This makes it an iterable dictionary
fn2class = dict(zip(df.index, df.pitches))
p_path = os.path.join("pickles", instrument + "_train.p")

#This pickle file must not be uploaded to git.
#This pickle contains the results from build_rand_feats in keyboard_model.py
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

#Loads the existing model (looks for the .pb file)
model=load_model(config.model_path)

testData = TestData(instrument)
test_labels, test_audio = build_predictions(audio_dir)
#This should print (numAudioFiles, 13, 9, 1)
test_audio = np.array(test_audio)
print(test_audio.shape)

score, acc = model.evaluate(test_audio, test_labels)
predict_labels = model.predict(test_audio)

count = 0
count021 = 0
total021 = 0
count2243 = 0
total2243 = 0
count4465 = 0
total4465 = 0
count6687 = 0
total6687 = 0
wrong = []
for i in range(0, predict_labels.shape[0]):
    if (np.argmax(np.array(test_labels[i])) <= 21):
        total021 += 1
    elif (np.argmax(np.array(test_labels[i])) <= 43 and np.argmax(np.array(test_labels[i])) >= 21):
        total2243 += 1
    elif (np.argmax(np.array(test_labels[i])) <= 65 and np.argmax(np.array(test_labels[i])) >= 44):
        total4465 += 1
    else:
        total6687 += 1
    if (np.argmax(predict_labels[i]) == np.argmax(np.array(test_labels[i]))):
        count += 1
    else:
        if (np.argmax(np.array(test_labels[i])) <= 21):
            count021 += 1
        elif (np.argmax(np.array(test_labels[i])) <= 43 and np.argmax(np.array(test_labels[i])) >= 21):
            count2243 += 1
        elif (np.argmax(np.array(test_labels[i])) <= 65 and np.argmax(np.array(test_labels[i])) >= 44):
            count4465 += 1
        else:
            count6687 += 1
        '''
        if (abs(np.argmax(predict_labels[i]) - np.argmax(np.array(test_labels[i])))>10):
            print(predict_labels[i][np.argmax(predict_labels[i])])
            print(np.argmax(np.array(test_labels[i])))
            print("")
        '''
        wrong.append(abs(np.argmax(predict_labels[i]) - np.argmax(np.array(test_labels[i]))))

print("Number correct: " + str(count))
print("Average distance from correct pitch: " + str(sum(wrong)/len(wrong)))
print("Wrong between 0 and 21: " + str(count021) + "/" + str(total021) + " = " + str(100*count021/total021) + "%")
print("Wrong between 22 and 43: " + str(count2243) + "/" + str(total2243) + " = " + str(100*count2243/total2243) + "%")
print("Wrong between 44 and 65: " + str(count4465) + "/" + str(total4465) + " = " + str(100*count4465/total4465) + "%")
print("Wrong between 66 and 87: " + str(count6687) + "/" + str(total6687) + " = " + str(100*count6687/total6687) + "%")

