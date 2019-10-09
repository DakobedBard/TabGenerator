from tabgen.jams_io import  jam_to_dataframes, tablature_dataframe
import librosa
import numpy as np
import sys
import pandas as pd
import os
'''
This file contains functions that are used to deal w/ generating, saving and loading the data. 

'''

def midi_to_matrix_index(midi_value):

    return int(midi_value-35)


def max_value_in_y():


    path = os.getcwd()
    path += '/data/spec_labels/train/'

    max_y = 0 

    for i, filepair in enumerate(annotation_audio_file_paths()):
      
        print(i, flush=True)
        newpath = path + 'fileid_'+str(i)+'/' 
        y,sr = librosa.load(filepair[0])
        maximum_y = np.max(y)
        if maximum_y > max_y:
            max_y = maximum_y
    return max_y


def annotation_audio_file_paths(audio_dir='data/audio/audio_mic', annotation_dir='data/annotation' ):

    audio_files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, file) for file in audio_files]

    annotation_files = os.listdir(annotation_dir)
    annotation_files = [os.path.join(annotation_dir, file) for file in annotation_files]

    #for file in annotation_files:
    #    print(file)

    file_pairs = []

    for annotation in annotation_files:
        annotation_file = annotation.split('/')[-1].split('.')[0]
        for audio_file in audio_files:
            if audio_file.split('/')[-1].split('.')[0][:-4] == annotation_file:
                file_pairs.append((audio_file, annotation))
                #print(audio_file, annotation_file)
    return file_pairs




def anonotation_matrix(annotation, song_duration, time_bins):
    ''' 
    This function will take an annotation and produce a vectorized annotation that will be used as a label.  
    '''
    # This line of code below belongs elsewhere I will move it some point.  
    
    time_slices = np.linspace(0,song_duration, time_bins)
    
    annotation['note_end'] = annotation['time'] + annotation['duration']

    annotation_array = np.zeros([48,time_bins])

    for index,row in annotation.iterrows():
        for i in range(time_slices.shape[0]):
            if row['time'] < time_slices[i]:
                note_start_index = i-1
            if row['note_end'] < time_slices[i]:
                note_end_index = i
                # Now fill in the values of of the annotation array.  This should be vectorized looping is stupid 
                for j in range(note_start_index, note_end_index):

                    midi_index = midi_to_matrix_index(row['value'])

                    annotation_array[midi_index][j] = 1
                break

    return annotation_array


def matrix_to_annotation(anno_mat):

    '''
    This function will be essentialy the inverse of the annotation_matrix function defined above.  Take a one hot encoded matrix for all 48 notes on the guitar and return an annotation.  A perfect inverse will not be possible of course...   

    '''
    columns = ['start','end', 'value']

    notes = []
    # Iterate through each note...
    is_note = False

    for i in range(anno_mat.shape[1]):

        # Iterate through each time slice

        for j in range(anno_mat.shape[0]):

            # Begininning of a new new note
            if anno_mat[j][i] == 1 and is_note == False:

                note_start = j
                is_note = True
                note_length = 0
            # Middle of note
            if anno_mat[j][i] ==1 and is_note == True:

                note_length +=1 
            # End of a note 
            elif anno_mat[j][i] ==0 and is_note == True:
                note_end = j
                # Only add a note if it is long enough...
                
                notes.append([.0232*note_start, .0232*note_end, i+35])

                is_note = False

    notes_df = pd.DataFrame(data = notes, columns = columns )
    notes_df.sort_values(by=['start'], inplace = True)
    notes_df = notes_df.reset_index(drop=True)

    return notes_df


def to_mat_to_df():
    '''
    This function will be used to compare the performance of going to annotation matrix and performing the inverse of this operation.  Will have to refactor later.  


    '''

    # filepair = annotation_audio_file_paths()[0]
    # y , sr = librosa.load(filepair[0])
    # annotation = tablature_dataframe(filepair[1])
    # annotation = annotation.reset_index(drop=True)

    # anno_mat = anonotation_matrix(annotation, y.shape[0]/sr, )
    # anno_mat = anno_mat.T

    # output_df = matrix_to_annotation(anno_mat)

    # return annotation, output_df

    pass







def save_transforms_and_annotations():

    '''
    Will possibly have to have some more advanced logic to handle the case where these directories do not exist yet... Such as when I move the training of the network to AWS.... 

    We are going to check if the file already exists in which case... we do not overwrite it.. 

    We are going to normalize the data ... 

    '''
    path = os.getcwd()
    path += '/data/spec_labels/train/'

    for i, filepair in enumerate(annotation_audio_file_paths()):
      
        print(i, flush=True)

        newpath = path + 'fileid_'+str(i)+'/' 
        
        if os.path.isdir(newpath) == False:
            os.mkdir(newpath)

        y,sr = librosa.load(filepair[0])
        # Try with just 4 octaves for the cqt... 
        fmin = librosa.note_to_hz('C2')
        cqt = np.abs(librosa.core.cqt(y,sr=sr,n_bins=120, bins_per_octave=24, fmin=fmin, norm=1)).T
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref = np.max).T

        annotation = tablature_dataframe(filepair[1])
        annotation = annotation.reset_index(drop=True)

        annotation_labels = anonotation_matrix(annotation, y.shape[0]/sr , spec.shape[0])
        annotation_labels = annotation_labels.T

        cqt_file = newpath+'cqt.npy'
        spec_file =newpath+'stft.npy'
        annotation_file = newpath+'annotation_label.npy'
        
        if os.path.isfile(cqt_file) == False:
            np.save(cqt_file, arr = cqt)
        if os.path.isfile(spec_file) == False:
            np.save(spec_file, arr= spec)
        if os.path.isfile(annotation_file) == False:
            np.save(annotation_file, arr = annotation_labels)
        
        if i >= 290:
           path = os.getcwd()
           path += '/data/spec_labels/test/'
        

def load_transform_and_annotation(id, spectogram = 'fft', train = True):


    path = os.getcwd()
    
    if train:
        path += '/data/spec_labels/train/fileid_'+str(id)+'/'
    else:
        path += '/data/spec_labels/test/fileid_'+str(id)+'/'

    annotation_label = np.load(path+'annotation_label.npy')

    if spectogram == 'fft':
        spec = np.load(path+'stft.npy')
        return spec, annotation_label
    elif spectogram == 'cqt':
        cqt = np.load(path+'cqt.npy')
        return cqt, annotation_label    


def load_train_data(number_of_songs, spectogram = 'fft'):
    '''
    Generate the training dataset that will be handed off to the model.  Will just concatenate the on axis  = 0 ...   

    '''
    path = os.getcwd()
    path += '/data/spec_labels/train/'

    for i in range(number_of_songs):

        spec , annotation = load_transform_and_annotation(i, spectogram=spectogram)
        if i == 0:
            X_train = spec
            y_train = annotation 
        else:

            return X_train, spec, y_train 
            X_train = np.concatenate((X_train, spec), axis = 0)
            y_train = np.concatenate((y_train, annotation), axis = 0)

    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train


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


def load_test_data():
    path = os.getcwd()
    path += '/data/spec_labels/test/'

    files = os.listdir(path)    
    min_file_index = min([int(f.split('_')[1]) for f in files])

    for i in range(len(files)):

        spec , annotation = load_transform_and_annotation(i+min_file_index, train=False)
        if i == 0:
            X_test = spec
            y_test = annotation 
        else:
            X_test = np.concatenate((X_test, spec), axis = 0)
            y_test = np.concatenate((y_test, annotation), axis = 0)


    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_test, y_test




