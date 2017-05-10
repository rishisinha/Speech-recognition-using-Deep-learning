import numpy as np
import tensorflow as tf
import pandas as pd

phn_data = pd.read_csv('phn_train.csv')
# mfcc_data = pd.read_csv('mfcc_train.csv')

phn_lib = ['aa','ae','ah','ao','aw','ax','ax-h','axr','ay','b',\
           'ch','d','dh','dx','eh','el','em','en','eng','er',\
           'ey','f','g','hh','hv','ih','ix','iy','jh','k','l',\
           'm','n','ng','nx','ow','oy','p','r','s','sh','t',\
           'th','uh','uw','ux','v','w','y','z','zh','bcl','dcl',\
           'gcl','kcl','pcl','qcl','tcl','q','epi','pau','h#']

#print len(phn_lib)
num_classes = len(phn_lib)

idx_to_phn = dict(enumerate(phn_lib))
phn_to_idx = dict(zip(idx_to_phn.values(),idx_to_phn.keys()))

print phn_to_idx
phn_all = np.asarray(phn_data['Phoneme'])
num_phns = phn_all.shape[0]
labels_one_hot = np.zeros((num_phns, num_classes))

x = []
for idx, label in enumerate(phn_all):
    if label != '--':
        x.append(phn_to_idx[label])
        labels_one_hot[idx][phn_to_idx[label]] = 1


df_targets = pd.DataFrame(labels_one_hot)
df_targets.to_csv('phn_target.csv')
# print len(labels_one_hot)
# print len(phn_data)
# print labels_one_hot[215:220][:]
