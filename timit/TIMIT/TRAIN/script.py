import glob
import librosa
import os

phn_files = glob.glob('*/*/*.PHN')
wav_files = glob.glob('*/*/*.WAV')

print len(wav_files)
mfcc_map = {}
for filename in wav_files:
    x,y = librosa.load(filename, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=16000)
    blah = filename.split('/' )
    mfcc_map[blah[-1][0:-4]] = mfcc

phn_map = {}

for filename in phn_files:
    with open(filename, 'r') as f:
        fcon = f.read()
        lines = fcon.split('\n')
        col = []
        for l in lines:
            col.append(l.split(' '))
    blah = filename.split('/' )
    phn_map[blah[-1][0:-4]] = col

