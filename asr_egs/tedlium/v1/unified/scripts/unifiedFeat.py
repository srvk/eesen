# -*- coding: utf-8 -*- 
import shutil  
import os  
import subprocess
from subprocess import Popen, PIPE, STDOUT
import shlex
import sys

original_wav = "../"
wav_path = "./"
praat_file = "praat.res"
final_file = "feat.txt"

# convert format of audio
# (no-op for TEDLIUM, already right format)
print "conversion started"
os.system("scripts/convert.sh")
print "conversion finished"
# duration, intensity, f0-f3, dir needs to be changed in the file
print "extraction started"

os.system("cp ./snr ./temp/")
#os.system("cp scripts/cat.py ./temp/")
os.system("cp ./feat.praat ./temp/")

os.chdir('./temp/')

# praat_path = script_path + "Syllable_test.praat"
praat_path = wav_path+"feat.praat"
step1 = "/usr/bin/praat " + praat_path
os.system("rm -rf ./praat.res")
Popen(step1, shell = True, stdout = PIPE).communicate()	

# /Applications/Praat.app/Contents/MacOS/Praat
# step1="/usr/bin/praat " + praat_path
# proc1 = subprocess.Popen(step1, shell=True)
# proc1.wait()

print "extraction finished"

# =================
#snr 
os.system("rm -rf ./snr.res")
print "snr started"
os.chdir(wav_path)
#proc2 = subprocess.Popen("for f in *.wav; do ./snr -SSNR -est1 $f \
#  	>> snr.res ; done", shell=True)

# for TEDLIUM audios, the above always gave -40.0000
# this gives more reasonable values
proc2 = subprocess.Popen("for f in *.wav; do ./snr -SSNR -estm -E $f \
  	>> snr.res ; done", shell=True)

proc2.wait()
print "snr finished"

# ===================
print "combinination started"
cat= "python ../scripts/cat.py praat.res snr.res feat.txt"

Popen(cat, shell = True, stdout = PIPE).communicate()

print "combinination finished"

os.system("cp ./feat.txt ../")


