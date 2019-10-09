from generate import generate_spectograms_annotations, generate_annotation_dataset 	
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, TimeDistributed, LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''

2018225 is how long the flattened spect

##  I suppose I could flatten the annotation array as well..
 
flatten the array w/ 

annotationfull_annotation_arrays_train.reshape((full_annotation_array.shape[0],48*1969))

reshape(full_annotation_array.shape[0], 48, 1969)    # 

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

'''

spectograms, annotations = generate_spectograms_annotations()

specs = spectograms[:len(annotations)]

full_annotation_array = generate_annotation_dataset(annotations)

specs_train, specs_test, annotations_train, annotations_test = train_test_split(specs, full_annotation_array)

print("Building model... ")




