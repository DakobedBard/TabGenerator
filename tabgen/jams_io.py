'''

This file contains functions which will parse the JAMS json data into pandas dataframes.  

'''



import os
import jams
import pandas as pd

def round_midi_numbers(midi):
    return round(midi)


def jam_to_dataframes(jam_file):

    ''' 
    This function will return a dictionary of dataframes for the given jam_file...
    '''

    annotation = (jams.load(jam_file)).annotations

    #annotation['value'] = annotation['value'].apply(round_midi_numers)

    lowE = annotation[1]
    A = annotation[3]
    D = annotation[5]
    G = annotation[7] 
    B = annotation[9]
    highE = annotation[11]
    
    return {1:lowE.to_dataframe(),2:A.to_dataframe(),3:D.to_dataframe(),
    4:G.to_dataframe(),5:B.to_dataframe(),6:highE.to_dataframe() }


def tablature_dataframe(jam_file):

    '''
    This function will take the dataframes from the jam_to_dataframes function and will concatenate it into a single dataframe.
    '''

    jams_dictionary = jam_to_dataframes(jam_file)

    for df,string_dict in jams_dictionary.items():
        string_dict['value'] = string_dict['value'].apply(round_midi_numbers)


    frames = []
    for dataframe in jams_dictionary:

        jams_dictionary[dataframe].drop('confidence', axis =1 , inplace = True)

        jams_dictionary[dataframe]['string'] = dataframe
        frames.append(jams_dictionary[dataframe])

    strings_df = pd.concat(frames)
    strings_df.sort_values('time', inplace=True)

    return strings_df




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