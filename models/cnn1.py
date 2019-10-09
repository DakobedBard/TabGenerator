import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from fit_generator import SpectogramAnnotationDataGenerator
from math import floor




'''
It is possible to provide my own metric, the classic metric is accuracy... 

I will have to wrap up some mir_eval functionality into a function which I will pass on to the model at the compilation step. 
Okay in order for this to work I think.. that I am going to have to add an extra dimension to the X_training data.... so that.. 

'''

    
num_epochs = 10

path = os.getcwd()
data_path = path + '/data/spec_labels/train/'

input_dimensions = (5,1025,1)


'''
    How do we determine the number of batches?


    382182 Training samples.. 

    88281 testing samples... 

'''

def specgenerator(spectogram = 'fft', batch_size = 32):

    generator = SpectogramAnnotationDataGenerator(batch_size=batch_size)
    yield generator.batch_generator()

def validation_generator(spectogram = 'fft', batch_size = 32):
    generator = SpectogramAnnotationDataGenerator(train=False, batch_size=batch_size)
    yield generator.batch_generator()



model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='normal', activation='relu', padding = 'same',input_shape=( 5,1025,1)))
model.add(MaxPool2D(pool_size =(2,2)))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(.2))
model.add(Dense(48,kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


batch_size = 32

model.fit_generator(generator=specgenerator(),
                                        epochs=num_epochs,
                                        steps_per_epoch = floor(382182/batch_size),
                                        verbose=1,
                                        use_multiprocessing=True,
                                        workers=16,
                                        validation_data = validation_generator(),
                                        validation_steps = floor(88281/batch_size),
                                        max_queue_size=32)
