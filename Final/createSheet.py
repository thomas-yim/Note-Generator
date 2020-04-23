import numpy as np
import mingus.extra.lilypond as lily
import mingus.containers as containers
import os
import subprocess

def save_string_and_execute_LilyPond(filename):
    """A helper function for to_png and to_pdf. Should not be used directly."""
    command = 'lilypond "%s.ly"' % (filename)
    print("Executing...")
    p = subprocess.Popen(command, shell=True).wait()
    return True

save_string_and_execute_LilyPond('test')

def find_key(song_notes):
    notes = np.zeros(12)
    for n in song_notes:
        if n != 0:
            notes[(n-21)%12] += 1
    sharps = 0
    flats = 0
    if notes[1] > notes[0] or notes[1] > notes[2]:
        if notes[0] > notes[2]:
            flats += 1
        else:
            sharps += 1
    if notes[4] > notes[3] or notes[4] > notes[5]:
        if notes[3] > notes[5]:
            flats += 1
        else:
            sharps += 1
    if notes[6] > notes[5] or notes[6] > notes[7]:
        if notes[5] > notes[7]:
            flats += 1
        else:
            sharps += 1
    if notes[9] > notes[8] or notes[9] > notes[10]:
        if notes[8] > notes[10]:
            flats += 1
        else:
            sharps += 1
    if notes[11] > notes[10] or notes[11] > notes[0]:
        if notes[10] > notes[0]:
            flats += 1
        else:
            sharps += 1
    sharp_keys = ['G', 'D', 'A', 'E', 'B', 'F#', 'C#']
    flat_keys = ['F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
    if sharps > flats:
        key = sharp_keys[sharps-1]
    elif flats > sharps:
        key = flat_keys[flats-1]
    else:
        key = 'C'
    return key

def create_sheet(song_notes, note_lengths, song_name='Untitled', fname='test'):
    key = find_key(song_notes)
    bars = [containers.Bar(key=key)]
    current_bar = 0
    bar_length = 0
    n = 0
    for note in song_notes:
        bar_length += 1/note_lengths[n]
        if bar_length > 1:
            bars.append(containers.Bar(key=key))
            current_bar += 1
            bar_length = 0
        if note != 0:
            bars[current_bar].place_notes(containers.Note().from_int(note - 12), note_lengths[n])
        else:
            bars[current_bar].place_rest(note_lengths[n])
        n += 1
    track = containers.Track()
    for bar in bars:
        track.add_bar(bar)
    comp = containers.Composition()
    comp.add_track(track)
    comp.set_title(song_name)
    lily_string = lily.from_Composition(comp)
    save_string_and_execute_LilyPond(lily_string, fname)
        
create_sheet(np.array(df['pitch']), np.array(df['length']))