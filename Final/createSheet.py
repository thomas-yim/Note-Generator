import numpy as np
import mingus.extra.lilypond as lily
import mingus.containers as containers
import subprocess
import songTranslator as translator
import sys
#The following code is modeled after the mingus.extra.lilypond library
def save_string_and_execute_LilyPond(lilyString, filename):
    file = open(filename + ".ly", "a")
    #The stuff added to the file is formatted for lilypond
    file.write(lilyString)
    print(filename)
    file.close()
    command = 'lilypond "%s.ly"' % (filename)
    print("Executing...")
    #This allows us to run a shell script from python
    p = subprocess.Popen(command, shell=True).wait()
    return True

"""
This returns the key of the song.
It basically does this by comparing the number of flats and sharps
"""
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

"""
This will actually create the pdf. It calls all the functions from previous files
It is basically our main() function
"""
def create_sheet(song_notes, note_lengths, song_name='Untitled', fname='test1'):
    last_note = -1
    #Our program checks for rests past the last note because the last note could
    #be a half note or longer. So, we have to delete rests after the last note "ends"
    for i in range(len(song_notes)-1,-1,-1):
        last_note = i
        if song_notes[i] > 0:
            break
        else:
            location = 0
            for note in note_lengths[:i+1]:
                location += 1/note
            if location % 1 == 0:
                break
    
    #The following code sets up the notes, durations, key, and rests in lilypond format
    key = find_key(song_notes)
    bars = [containers.Bar(key=key)]
    current_bar = 0
    bar_length = 0
    for i in range(0, last_note+1):
        note = song_notes[i]
        if bar_length >= 1:
            bars.append(containers.Bar(key=key))
            current_bar += 1
            bar_length = 0
        if note != 0:
            print(bars[current_bar].place_notes(containers.Note().from_int(note - 12), note_lengths[i]))
        else:
            bars[current_bar].place_rest(note_lengths[i])
        bar_length += 1/note_lengths[i]
    track = containers.Track()
    print(bars)
    for bar in bars:
        track.add_bar(bar)
    comp = containers.Composition()
    comp.add_track(track)
    comp.set_title(song_name)
    lily_string = lily.from_Composition(comp)
    save_string_and_execute_LilyPond(lily_string, song_name)
        
#This is so we can run the program like python3 createSheet.py instrument filename name
#If not, you can just input the values when run.
#The sys.argv is mainly for the server use.    
if len(sys.argv) ==  4:
    inst = sys.argv[1]
    fname = sys.argv[2]
    pdfName = sys.argv[3]
elif len(sys.argv) == 3:
    inst = sys.argv[1]
    fname = sys.argv[2]
    pdfName = sys.argv[2]
else:
    inst = input("What type of instrument (keyboard or guitar) is it? ").lower()
    fname = input("What is the name of the file? ")
    pdfName = input("What would you like to name the song? ")
df = translator.recognizeSong(fname, inst)
create_sheet(np.array(df['type']), np.array(df['length']), song_name=pdfName, fname=fname)
