import pandas as pd

a = pd.read_csv('phn_train.csv')
b = pd.read_csv('mfcc_train.csv')

c = pd.read_csv('phn_target.csv')

# print merger.ix[:,2:15]
mfcc_normalise = pd.DataFrame(columns = ('MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13'))

for i in range(len(b)):
    b.ix[i,2:15] = b.ix[i,2:15] - min(b.ix[i,2:15])
    b.ix[i,2:15] = b.ix[i,2:15]/(b.ix[i,2:15]).sum()

mfcc_normalise = b.ix[:,1:15]

c['sum_of_coeffs'] = 0

for i in range(len(a)):
    if a.ix[i,'Phoneme'] == '--':
        mfcc_normalise.ix[i,'File'] = 0


for i in range(len(c)):
    c.ix[i,'sum_of_coeffs'] = c.ix[i,1:].sum()

inputs = mfcc_normalise[mfcc_normalise.File != 0]
outputs = c[c['sum_of_coeffs'] == 1]

inputs.to_csv('inputs.csv')
outputs.to_csv('outputs.csv')
