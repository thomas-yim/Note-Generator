'''
Some notes:
 - This algorithm is based on the findings of the following research paper:
   https://www.ee.columbia.edu/~dpwe/papers/Laro01-swing.pdf
 - In this version, the BPM will be most accurately estimated if the piece
   is in 4/4 time (but it will be alright for 2/4 and 2/2 as well)
'''

import math
import numpy as np

'''
returns the predicted probability at the given location according to an
adjusted version of the normal distribution function
'''
def pdf(var, shift, value):
    return (1/math.sqrt(2*math.pi*var)) * math.exp(-0.5*math.pow((value-shift)/var, 2))

'''
returns the predcited probability that a note will begin at the given location
'''
def transient_probability(time, period):
    # adjusts the time of the note to be within the period
    timeP = time%period
    # returns a value from 0-0.4 as the probability that a note will fall at the given
    # location in the period
    return 0.4 * pdf(0.00125*period, 0, timeP) + 0.15 * pdf(0.00125*period, period/4, timeP) + 0.3 * pdf(0.00125*period, period/2, timeP) + 0.15 * pdf(0.00125*period, 3*period/4, timeP) + 0.4 * pdf(0.00125*period, period, timeP)

'''
returns the calculated tempo of a piece of music given an array of the indexes
in the signal array that each note starts and the signal rate
'''
def get_bpm(note_starts, sr):
    # converts array to time in seconds
    note_times = (np.asarray(note_starts)-note_starts[0])/sr
    # adjusts the times to begin on the first note
    adj_times = np.asarray(note_times) - note_times[0]
    # array that will populate with the likelihood that a tempo is the correct tempo
    likelihood = np.zeros(70)
    # searches for a tempo from 70 to 139 BPM
    for t in range(70,140):
        # period is the length of a bar in 4/4 time in seconds
        period = 240/t
        for n in adj_times:
            # adds up a likelihood score for each tempo based on how likely it
            # would be for a note to begin at the given time in this tempo
            likelihood[t-70] += (transient_probability(n, period))
    # returns the tempo with the maximum likelihood score
    return np.argmax(likelihood, axis=None)+70

'''
returns an array of the classifications of each note in a piece of music as an 
eighth, quarter, half, etc. note given an array of when each note starts,
an array of when each note stops, the signal rate and the tempo of the piece
'''
def classify_note_types(note_starts, note_ends, sr, bpm):
    # defined to help with naming
    note_classes = ["sixteenth", "eighth", "quarter", "half", "full"]
    # converts arrays to time in seconds
    start_times = (np.array(note_starts)-note_starts[0])/sr
    end_times = (np.array(note_ends)-note_starts[0])/sr
    # duration of a quarter note
    q_dur = 60/bpm
    # array that will fill with the note classifications
    n_types = []
    for i in range(len(start_times)):
        # calculates the index of the correct classification name in the note_classes array
        type_index = int(round(math.log((end_times[i] - start_times[i])/q_dur,2) + 2))
        # adds the note classification to the array that will be returned
        try:
            n_types.append(note_classes[type_index])
        except Exception:
            if type_index < 0:
                n_types.append("short")
            else:
                n_types.append("long")
    return n_types