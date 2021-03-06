import glob
import librosa
import os
import pandas as pd
import numpy as np
import python_speech_features as psf

phn_files = glob.glob('*/*/*.PHN')
wav_files = glob.glob('*/*/*.WAV')

# x_,y_ = librosa.load('DR1/FCJF0/SX37.WAV', sr=16000)

# print len(x_)
# mfcc = psf.mfcc(x_, winlen = 0.016, winstep = 0.008)
# print mfcc.shape





win_size = 256  #512
hop_size = 128  #256

mfcc_map = {}
phn_map = {}

for filename in wav_files[:250]:
    x,y = librosa.load(filename, sr=16000)
    # print(len(x))
    # mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13, n_fft=win_size, hop_length = hop_size)
    mfcc = psf.mfcc(x, winlen = 0.016, winstep = 0.008, nfft = 256)
    mfcc = mfcc.T
    blah = filename.split('/' )
    mfcc_map[blah[-3] + blah[-2] + blah[-1][0:-4]] = mfcc


#for filename in phn_files[:1]:
    filename_ = blah[-3] + '/' + blah[-2] + '/' + blah[-1][0:-4] + '.PHN'
    with open(filename_, 'r') as f:
        fcon = f.read()
        lines = fcon.split('\n')
        col = []
        for l in lines:
            col.append(l.split(' '))
    blah = filename.split('/' )
    phn_map[blah[-3] + blah[-2] + blah[-1][0:-4]] = col

df_phn = pd.DataFrame(columns = ('File', 'Phoneme'))
# x = []
q = 0

for key, val in phn_map.iteritems():
    if key == 'DR1FDAW0SI1271':
        continue
    num_samples = (mfcc_map[key].shape[1])*hop_size
    # print mfcc_map[key].shape
    i = 0
    while i < num_samples:
        for j in range(len(phn_map[key])-1):
            if i >= int(phn_map[key][j][0]) and i + win_size <= int(phn_map[key][j][1]):
                # x.append(phn_map[key][j][2])
                df_phn.loc[q] = [key, phn_map[key][j][2]]
                q += 1
                break

            elif i >= int(phn_map[key][j][0]) and i <= int(phn_map[key][j][1]):
                # x.append('--')
                df_phn.loc[q] = [key, '--']
                q += 1
                break
        i += hop_size

df_mfcc = pd.DataFrame(columns = ('File', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13'))

w = 0

for key, val in mfcc_map.iteritems():
    if key == 'DR1FDAW0SI1271':
        continue
    for k in range(val.shape[1]):
        df_mfcc.loc[w] = [key, val[0,k], val[1,k], val[2,k], val[3,k], val[4,k], val[5,k], val[6,k], val[7,k], val[8,k], val[9,k], val[10,k], val[11,k], val[12,k]]
        w += 1

print len(df_mfcc)
print len(df_phn)

#for key, val in phn_map.iteritems():
#    print key
#    b = df_mfcc[df_mfcc.File == key].count() 
#    c = df_phn[df_phn.File == key].count()
#    print b
#    print c

a = df_mfcc.loc[(df_mfcc['File'] != df_phn['File'])].index
print len(a)
df_mfcc.to_csv('mfcc_train.csv')
df_phn.to_csv('phn_train.csv')
# DR1FDAW0SI1271
