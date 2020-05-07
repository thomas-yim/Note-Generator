import os
import subprocess

#The following code is modeled after the mingus.extra.lilypond library
def save_string_and_execute_LilyPond(ly_string, filename):
    """A helper function for to_png and to_pdf. Should not be used directly."""
    command = 'lilypond "%s.ly"' % (filename)
    print("Executing...")
    p = subprocess.Popen(command, shell=True).wait()
    return True

save_string_and_execute_LilyPond('test', 'test')