import glob
import librosa
import os
import pandas as pd
import numpy as np

phn_files = glob.glob('*/*/*.PHN')
wav_files = glob.glob('*/*/*.WAV')

# x_,y_ = librosa.load('DR1/FCJF0/SX37.WAV', sr=16000)

win_size = 512
hop_size = 256

mfcc_map = {}
for filename in wav_files[:200]:
    x,y = librosa.load(filename, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13, n_fft=win_size, hop_length = hop_size)
    blah = filename.split('/' )
    mfcc_map[blah[-3] + blah[-2] + blah[-1][0:-4]] = mfcc

phn_map = {}

for filename in phn_files[:200]:
    with open(filename, 'r') as f:
        fcon = f.read()
        lines = fcon.split('\n')
        col = []
        for l in lines:
            col.append(l.split(' '))
    blah = filename.split('/' )
    phn_map[blah[-3] + blah[-2] + blah[-1][0:-4]] = col

df_phn = pd.DataFrame(columns = ('File', 'Phoneme'))
x = []
q = 0

for key, val in phn_map.iteritems():

    num_samples = (mfcc_map[key].shape[1] - 1)*hop_size
    i = 0
    while i < num_samples:
        for j in range(len(phn_map[key])-1):
            if i >= int(phn_map[key][j][0]) and i + win_size < int(phn_map[key][j][1]):
                x.append(phn_map[key][j][2])
                df_phn.loc[q] = [key, phn_map[key][j][2]]
                q += 1
                break

            elif i >= int(phn_map[key][j][0]) and i < int(phn_map[key][j][1]):
                x.append('--')
                df_phn.loc[q] = [key, '--']
                q += 1
                break
        i += hop_size

df_phn.to_csv('phn_train.csv')
