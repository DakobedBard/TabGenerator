
'''
Make use of the keras fit_generator API to process the data in batches. 


Let ID be the ython string that identifies a given sample of the dataset.  


'''


import numpy as np
import os
from random import shuffle
from keras.utils import Sequence
from keras.callbacks import Callback

'''
At the end of the epoch... I will have to reshuffle.  Find out which method i have to implement to be called at the end of the eoch..

'''



def generator():
    i = 0 
    while True:
        i += 1
        yield i


for item in generator():
    print(item)
    if item >4:
        break





class AccuracyHistory(Callback):

    def on_train_begin(self, logs = {}):
        self.acc = []
    def on_epoch_end(self,batch, logs = {}):
        self.acc.append(logs.get('acc'))

class SpectogramAnnotationBatch():
    '''
    I should consider creating a class for the batches... that might make working with them easiser. 
    
    I could 
    
    '''
    def __init__(self,dimensions):
        self.dimensions = dimensions


def SpectogramGenerator(batch_size = 32, batch_dimension =(5,1025), spectogram = 'fft', train = True):
    '''


    '''

    if train:

        fileids = list(range(5))
    else:
        fileids = [f +291 for f in range(5)]

    shuffle(fileids)
        
    spectogram_queue = fileids
    current_spectogram_index = 0 
    batch_number = 0
    current_songid = spectogram_queue.pop(0)
    X_train, y_train = load_transform_and_annotation(current_songid) 
    while True:


        if len(spectogram_queue) >0:
            current_song_id = spectogram_queue.pop(0)
        else:
            raise StopIteration
        if current_spectogram_index + batch_size >= X_train.shape[0]:
            if len(spectogram_queue) >0:                
            
                current_songid  = spectogram_queue.pop(0)
            else:
                raise StopIteration
            next_spec , next_annotation = load_transform_and_annotation(current_songid , train=train)
            
            BatchX,Batchy, X_train, y_train, current_spectogram_index = stitch( next_spec, next_annotation, batch_size, X_train, y_train, current_spectogram_index)
                
            newdim = [batch_size,*batch_dimension,1]

            BatchX = BatchX.reshape(newdim)
            batch_number += 1
            print(batch_number, flush = True)
            yield BatchX, Batchy
        else:

            BatchX = X_train[current_spectogram_index:(current_spectogram_index+batch_size)]
            Batchy = y_train[current_spectogram_index:(current_spectogram_index+batch_size)]

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



    def batch_generator(self):
        '''
        Generate one batch of data
        I will have an array of the currently being spectograms and annotations... when stitch gets called, self.X_train and self.y_train will be reassinged with the spectra for the new dataset... This function will also return the batches..
        
        the __iter__ methhod will yield the training and annotations..
        
        
        '''
        self.current_song_id = self.spectogram_queue.pop(0)
        self.X_train, self.y_train = self.load_transform_and_annotation(self.current_song_id)
        self.current_spectogram_index = 0   
        batch_number = 0
        while True:
            ## Loop back around...
            # Pop songs off the queue...
            # This is the corner case where we have to stitch together the spectra..
            print(self.current_spectogram_index)
            if self.current_spectogram_index + self.batch_size >= self.X_train.shape[0]:
                print(self.current_spectogram_index)
                if len(self.spectogram_queue) ==0:
                    self.init_spec_queue()

                self.current_song_id  = self.spectogram_queue.pop(0)

                next_spec , next_annotation = self.load_transform_and_annotation(self.current_song_id)
            
                BatchX,Batchy = self.stitch( next_spec, next_annotation)
                
                newdim = [self.batch_size,*self.batch_dimension,1]

                BatchX = BatchX.reshape(newdim)
                print(batch_number, flush = True)

                yield BatchX, Batchy 





class SpectogramAnnotationDataGenerator(Sequence):

    '''
    382182

    88281

    '''


    def __init__(self, batch_size=32,batch_dimension = (5,1025), n_channels=1, shuffle=True, train = True, spectogram = 'fft'):
    
        self.batch_size = batch_size
        self.n_channels = n_channels
        
        self.shuffle = shuffle
        
        self.train = train
        self.spectogram = spectogram
        self.path  =os.getcwd()
        if self.train:
            self.path += '/data/spec_labels/train/'
            self.num_batches = 382182 //self.batch_size
        else:
            self.path += '/data/spec_labels/test/'
            self.num_batches = 88281//self.batch_size
        
        files = os.listdir(self.path)

        self.nfiles = len(files)

        self.batch_size = batch_size
        self.batch_dimension = batch_dimension
        self.init_spec_queue()



    def on_epoch_end(self):
        self.init_spec_queue()
    def init_spec_queue(self):

        if self.train:

            fileids = list(range(5))
        else:
            fileids = [f +291 for f in range(5)]

        shuffle(fileids)
        
        self.spectogram_queue = fileids



    def __getitem__(self,idx):
        pass
        


    def load_transform_and_annotation(self, id, train = True):

        path = self.path + 'fileid_' + str(id) + '/'

        annotation_label = np.load(path+'annotation_label.npy')

        if self.spectogram == 'fft':
            spec = np.load(path+'stft.npy')

            X_train = self.generate_windowed_samples(spec)


            return X_train, annotation_label
        elif self.spectogram == 'cqt':
            cqt = np.load(path+'cqt.npy')

            X_train = self.generate_windowed_samples(cqt)

            return X_train, annotation_label    

    def generate_windowed_samples(self,spec):
        '''
        Alright I believe that we will be having windows... 

        '''
        windowed_samples =np.zeros((spec.shape[0],5, spec.shape[1]))
        for i in range(spec.shape[0]):
            windowed_samples[i] = self.generate_sample(spec,i)

            ## The pointer logic would go here it seems like... 
            #windowed_samples[i] = spec[i -2:i +3]
        return windowed_samples


    def generate_sample(self,spec, time_index):
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






    def batch_generator(self):
        '''
        Generate one batch of data
        I will have an array of the currently being spectograms and annotations... when stitch gets called, self.X_train and self.y_train will be reassinged with the spectra for the new dataset... This function will also return the batches..
        
        the __iter__ methhod will yield the training and annotations..
        
        
        '''
        self.current_song_id = self.spectogram_queue.pop(0)
        self.X_train, self.y_train = self.load_transform_and_annotation(self.current_song_id)
        self.current_spectogram_index = 0   
        
        while True:
            ## Loop back around...
            # Pop songs off the queue...
            # This is the corner case where we have to stitch together the spectra..
            print(self.current_spectogram_index)
            if self.current_spectogram_index + self.batch_size >= self.X_train.shape[0]:
                print(self.current_spectogram_index)
                if len(self.spectogram_queue) ==0:
                    self.init_spec_queue()

                self.current_song_id  = self.spectogram_queue.pop(0)

                next_spec , next_annotation = self.load_transform_and_annotation(self.current_song_id)
            
                BatchX,Batchy = self.stitch( next_spec, next_annotation)
                
                newdim = [self.batch_size,*self.batch_dimension,1]

                BatchX = BatchX.reshape(newdim)

                yield BatchX, Batchy 
            else:
                
                BatchX = self.X_train[self.current_spectogram_index:(self.current_spectogram_index+self.batch_size)]
                Batchy = self.y_train[self.current_spectogram_index:(self.current_spectogram_index+self.batch_size)]

                newdim = [self.batch_size,*self.batch_dimension,1]
                BatchX = BatchX.reshape(newdim)
             
                yield BatchX, Batchy


    def stitch(self,  next_spec, next_annotation):

        ''' 
        Calculate how many samples of the next spectogram I need to grab. Then set the current_spectogra_index to this value             
        This method will be called when the spectogram gets pulled off the queue requiring the need to stitch together the spectograms

    
        '''
        n_samples = self.batch_size+self.current_spectogram_index-self.X_train.shape[0]
        prev_n_samples = self.batch_size - n_samples

        spec1 = self.X_train[-prev_n_samples:]
        spec2 = next_spec[:n_samples]
        BatchX = np.concatenate((spec1,spec2),axis=0)
        annotation1 = self.y_train[-prev_n_samples:]
        annotation2 = next_annotation[:n_samples]
        Batchy = np.concatenate((annotation1,annotation2), axis =0) 

        self.X_train = next_spec
        self.y_train = next_annotation
        self.current_spectogram_index = n_samples

        return BatchX, Batchy

        
    def stitch_spectra(self, current_spec, next_spec,current_annotation, next_annotation, n_stiched_samples):
        '''
        This method will be called when we get to the end of one spectra, and pop off a spectra from the queue
        Should return a Batch... so something of size batchsize, *dimensions

        '''
        StichedX = np.zeros((self.batch_size,*self.batch_dimension))
        StitchedAnnotation = np.zeros((self.batch_size,48))
        StichedX[: self.batch_size - n_stiched_samples] = current_spec[-(self.batch_size-n_stiched_samples):]
        StichedX[:n_stiched_samples] = next_spec[:n_stiched_samples]
        StitchedAnnotation[:self.batch_size - n_stiched_samples] =  current_annotation[-(self.batch_size-n_stiched_samples):]

        return StichedX, StitchedAnnotation


    def __len__(self):

        return self.num_batches

