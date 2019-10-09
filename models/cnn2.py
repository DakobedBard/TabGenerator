


import os
from generate import load_train_data, load_test_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D
from fit_generator import SpectogramAnnotationDataGenerator



def train_cnn():
    
    num_epochs = 10

    path = os.getcwd()
    data_path = path + '/data/spec_labels/train/'

    generator = SpectogramAnnotationDataGenerator(spectogram='cqt')
    validation_generator = SpectogramAnnotationDataGenerator(train=False,spectogram='cqt')






    # model = Sequential()
    # model.add(Conv2D(48,(3,3),  kernel_initializer='normal', activation='sigmoid', padding = 'same',input_shape=(5,1025,1)))

    # model.add(Dropout(.2))

    # model.add(Dense(48,kernel_initializer='normal', activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    
    # model.fit_generator(generator=generator,
    #                                     epochs=num_epochs,
    #                                     verbose=1,
    #                                     use_multiprocessing=True,
    #                                     workers=16,
    #                                     steps_per_epoch=int((1000000 // generator.batch_size)),
    #                                     max_queue_size=32)








