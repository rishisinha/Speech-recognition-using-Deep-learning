import pandas as pd

a = pd.read_csv('phn_train.csv')
b = pd.read_csv('mfcc_train.csv')

print a.head(50)
# print len(a)
#print len(a.Phoneme.unique())