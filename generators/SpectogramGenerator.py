import numpy as np
import os
from random import shuffle
from keras.utils import Sequence
from keras.callbacks import Callback



def testGenerator():


    i = 0
    generator = SpectogramGenerator()
    while True:

        i += 1
        BatchX,Batchy = next(generator)
        print(X.shape)
        
        
def shuffle_files(train = True):        
    if train:

        fileids = list(range(5))
    else:
        fileids = [f +291 for f in range(5)]
    
    shuffle(fileids)
    return fileids
    
def SpectogramGenerator(batch_size = 32, batch_dimension =(5,1025), spectogram = 'fft', train = True):
    '''

    '''



    spectogram_queue = shuffle_files(train=train)
    current_spectogram_index = 0 
    batch_number = 0
    current_songid = spectogram_queue.pop(0)
    X_train, y_train = load_transform_and_annotation(current_songid, train=train) 
    while True:

        if len(spectogram_queue) <= 0:
            spectogram_queue = shuffle_files(train=train)
            current_songid = spectogram_queue.pop(0)
        
        #print(current_song_id, flush=True)
        if current_spectogram_index + batch_size >= X_train.shape[0]:
            current_songid = spectogram_queue.pop(0)
            next_spec , next_annotation = load_transform_and_annotation(current_songid , train=train)
    
            BatchX,Batchy, X_train, y_train, current_spectogram_index = stitch( next_spec, next_annotation, batch_size, X_train, y_train, current_spectogram_index)
                
            newdim = [batch_size,*batch_dimension,1]

            BatchX = BatchX.reshape(newdim)
            batch_number += 1
            #print(batch_number, flush = True)
            yield BatchX, Batchy
        else:
            BatchX = X_train[current_spectogram_index:(current_spectogram_index+batch_size)]
            Batchy = y_train[current_spectogram_index:(current_spectogram_index+batch_size)]
            current_spectogram_index = current_spectogram_index + batch_size
            newdim = [batch_size,*batch_dimension,1]
            BatchX = BatchX.reshape(newdim)
             
            yield BatchX, Batchy


def load_transform_and_annotation(id, train = True, spectogram = 'fft'):
    if train:
        path = 'data/spec_labels/train/' + 'fileid_' + str(id) + '/'
    else:
        path = 'data/spec_labels/test/'+'fileid_' + str(id) + '/'
    annotation_label = np.load(path+'annotation_label.npy')

    if spectogram == 'fft':
        spec = np.load(path+'stft.npy')

        X_train = generate_windowed_samples(spec)

        return X_train, annotation_label
    elif spectogram == 'cqt':
        cqt = np.load(path+'cqt.npy')

        X_train = generate_windowed_samples(cqt)

        return X_train, annotation_label    


def generate_windowed_samples(spec):
    '''
    Alright I believe that we will be having windows... 

    '''
    windowed_samples =np.zeros((spec.shape[0],5, spec.shape[1]))
    for i in range(spec.shape[0]):
            windowed_samples[i] = generate_sample(spec,i)

            ## The pointer logic would go here it seems like... 
            #windowed_samples[i] = spec[i -2:i +3]
    return windowed_samples


def generate_sample(spec, time_index):
    '''
    This function will two take an index... 

    '''

    if time_index <= 1:
        slice_window = np.zeros((5, spec.shape[1]))
    elif time_index >= spec.shape[0]-2:
        slice_window = np.zeros((5, spec.shape[1]))

    else:
        slice_window = spec[time_index-2:time_index+3]
    return slice_window



def load_train_data(dimension=(5,1025)):
    

    
    
    Xtrain = np.zeros(5, *dimension, 382182)
    y_train = np.zeros(48,382182)
    
def load_test_data(dimension=(5,1025)):
    X_test = np.zeros(5, *dimension, 88281)
    y_test = np.zeros(48, 88281)




def stitch(next_spec, next_annotation, batch_size, X_train,y_train, current_spectogram_index):

    ''' 
        Calculate how many samples of the next spectogram I need to grab. Then set the current_spectogra_index to this value             
        This method will be called when the spectogram gets pulled off the queue requiring the need to stitch together the spectograms

    
    '''
    n_samples = batch_size+current_spectogram_index-X_train.shape[0]
    prev_n_samples = batch_size - n_samples

    spec1 = X_train[-prev_n_samples:]
    spec2 = next_spec[:n_samples]
    BatchX = np.concatenate((spec1,spec2),axis=0)
    annotation1 = y_train[-prev_n_samples:]
    annotation2 = next_annotation[:n_samples]
    Batchy = np.concatenate((annotation1,annotation2), axis =0) 

    X_train = next_spec
    y_train = next_annotation
    current_spectogram_index = n_samples

    return BatchX, Batchy, X_train, y_train, current_spectogram_index