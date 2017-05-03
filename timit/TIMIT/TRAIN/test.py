import pandas as pd

a = pd.read_csv('phn_train.csv')
b = pd.read_csv('mfcc_train.csv')

c = pd.read_csv('phn_target.csv')
print c['12'].sum()
# print len(a)
#print len(a.Phoneme.unique())