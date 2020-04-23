import os
import subprocess

def save_string_and_execute_LilyPond(filename):
    """A helper function for to_png and to_pdf. Should not be used directly."""
    command = 'lilypond "%s.ly"' % (filename)
    print("Executing...")
    p = subprocess.Popen(command, shell=True).wait()
    return True

save_string_and_execute_LilyPond('test')