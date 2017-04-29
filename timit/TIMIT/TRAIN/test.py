import pandas as pd

a = pd.read_csv('phn_train.csv')
print len(a.Phoneme.unique())
