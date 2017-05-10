# Speech-recognition-using-Deep-learning

This is a project to implement speech recognition using Deep learning. It performs speech recognition by detecting the phoneme uttered at each frame.

## Files

These files needs to be placed within the TRAIN and TEST folders of the TIMIT folder. Changes need to be made to specify whether the the generated data is training or testing data, in each folder.

### script.py

This is used for getting the MFCC information from the audio files. The MFCCs in each frame are mapped to the corresponding phoneme from the .PHN files. In case of ambiguity in the phoneme mapping, the corresponding frame is dropped. The MFCCs and phonemes are stored in the form of a Pandas table.

### one_hot.py

This is used to one hot encode the target phonemes.

### test.py

This normalizes the MFCCs across the 13 MFCC feature points. It does so by subtracting the mean and dividing by the standard deviation. This also drops the frames which correspond to more than one phoneme.

### rnn.py

This is the file used for training the network, and testing on the test dataset.


## Dependencies

Python
Numpy
Pandas
Python Speech Features
Tensorflow
Librosa
