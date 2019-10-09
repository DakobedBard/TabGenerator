
'''
This file defines a Spectogram class.

'''

import librosa
import librosa.display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jams
import plotly 
import plotly.graph_objs as go

from tabgen.sounds import play_audio

import scipy.signal


import mir_eval



class Spectogram:

    def __init__(self, file_id, win = 8820, step = 4410):

        '''
        This function will initialize a spectogram object from a file ID.  This will be used to initialize spectograms for the training data
        '''

        if isinstance(file_id, int):
            self.init_from_fileid(file_id)

        if isinstance(file_id, str):
            self.init_from_wav(file_id)


    def init_from_fileid(self, file_id):

        files = annotation_audio_file_paths()
        self.y, self.sr = librosa.load(files[file_id][0])
        fname = files[file_id][0].split('.')[0].split('/')[-1]
        self.D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref = np.max)
        self.S = librosa.stft(self.y)

        #self.Sp,self.t,self.f = pyaudio_spec(self.y, self.sr, win = win, step = step)         
        self.bpm, self.beats = extract_tempo_and_beats(self.y, self.sr)
        self.name = fname

        audio_dir = 'data/audio/audio_mic/'

        self.wav_file_path = audio_dir+self.name+'.wav'

        self.tempo = int(fname.split('-')[1])

        self.annotation = tablature_dataframe(files[file_id][1])

       
        self.o_env = librosa.onset.onset_strength(self.y, self.sr)


    def init_from_wav(self, wav_file_path):

        '''
        This function will initialize a Spectogram object from a wav file. 
        '''

        self.wav_file_path = wav_file_path

        self.name = wav_file_path.split('/')[-1].split('.')[0]

        self.y, self.sr = librosa.load(wav_file_path)

        self.D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref = np.max)
        self.S = librosa.stft(self.y)

        self.bpm, self.beats = extract_tempo_and_beats(self.y, self.sr)
        self.o_env = librosa.onset.onset_strength(self.y, self.sr)


    def play_wav_audio(self):

        '''
        This function plays the wav file for this Spectogram object
        '''

        play_audio(self.wav_file_path)

    def plot_chroma_stft(self): 
        '''
        A chroma vetor is typically a 12-element feature vector indicating how much energy
        of each pitch clas {C,C#, D,D#,E,F,F#,G,G#,A,A#,B} is present in the signal

        '''
        fmin = librosa.midi_to_hz(36)
        hop_length = 512
        C = librosa.core.cqt(self.y, sr = self.sr, fmin = fmin, n_bins = 72, hop_length=hop_length)


        logC = librosa.amplitude_to_db(np.abs(C))
        
        chromagram = librosa.feature.chroma_stft(self.y, sr = self.sr,hop_length=hop_length)
        plt.figure(figsize=(15,5))

        librosa.display.specshow(chromagram, x_axis ='time', y_axis='chroma', hop_length=hop_length, cmap ='coolwarm')
        plt.show()

    def plot_chroma_cqt(self, plot = False):
        fmin = librosa.midi_to_hz(36)
        hop_length = 512
        C = librosa.core.cqt(self.y, sr = self.sr, fmin = fmin, n_bins = 72, hop_length=hop_length)
        chromagram = librosa.feature.chroma_cqt(self.y, sr = self.sr,hop_length=hop_length)
        
        if plot:
        
            librosa.display.specshow(chromagram, x_axis ='time', y_axis='chroma', hop_length=hop_length, cmap ='coolwarm')
            plt.show()

        return chromagram



    def tohz(self, note):

        return librosa.midi_to_hz(note)



    def annotation_hz_column(self):

        '''
        This method will apply a transformation on the annotations datarame mapping from midi numbers to hz
        '''
        annotations = self.annotation

        annotations['hz'] = annotations['value'].apply(self.tohz)

        return annotations

    
    def transcription_evaluation(self):

        '''
        This method will use mir_eval to evaluate the transcription.

        I now have a method that returns the onset off set times and frequenices, for each note.  These are the estiamte notes

        Next I have to get the annotated 

        '''
     
        reference_onsets = self.annotation['time'].values
        reference_offsets = (self.annotation['time']+self.annotation['duration']).values
        reference_freq = self.annotation['value'].values

        reference_freq = librosa.note_to_hz(librosa.midi_to_note(reference_freq))

        ref_intervals = np.zeros((reference_freq.shape[0],2))
        ref_intervals[:,0] = reference_onsets

        ref_intervals[:,1] = reference_offsets

        onnset_offset_freq  = self.generate_estimated_annotation()

        estimated_onsets = onnset_offset_freq[:,0]
        estiamted_offsets = onnset_offset_freq[:,1]
        estimated_pitches = onnset_offset_freq[:,2]


        estimated_intervals = np.zeros((estimated_onsets.shape[0],2))
        estimated_intervals[:,0] = estimated_onsets

        estimated_intervals[:,1] = estiamted_offsets

        ### Great now we should be able to evaluate the transcription using mir_Eval... 

        precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, reference_freq, estimated_intervals, estimated_pitches)

        return (precision, recall)

        #return (reference_onsets, reference_offsets, reference_freq, estimated_onsets, estiamted_offsets, estimated_pitches) 


    def count_unique_annotated_notes(self):


        '''
        This method was written in order to attempt to evaluate the annotations.. 

        returns a set of intervals in which the times between two successive annotation time stampts 
        are less than a threshold, by deaflt =.05.  I will assume that these annotations will be of chords...
        next I need to return some information about the strings on which this is occuring.. 

        ask about his tommorow..

        '''

        note_count = 0

        intervals = []

        times = self.annotation_onsets(units='time')

        for i in range(1,times.shape[0]-1):
            time_diff = times[i] - times[i-1]     
            if time_diff >.05:
                note_count += 1    
            else:
                intervals.append((times[i-1],times[i]))

        return (note_count, intervals)


    def annoated_chords(self):

        '''
        This method attempts to find the chords (or double stops) in the annotation.

        This method will return pairs of times of notes that occur within .05 seconds of each other as well
        as the string that each of these pairs of notes are being played on.  

        '''

        chords = []

        annotation_values = self.annotation.values

        for i in range(1, annotation_values.shape[0]):
            
            time_diff = annotation_values[i,0] - annotation_values[i-1, 0]
            
            if time_diff < 0.05:
                chords.append((annotation_values[i-1,0], annotation_values[i, 0], annotation_values[i-1,3], annotation_values[i, 3]))
            
        return chords



    def annotation_onsets(self, units = 'time'):
        
        if units == 'frames':
        
            return librosa.time_to_frames(times = self.annotation['time'].values, sr = self.sr)

        if units == 'time':
            return self.annotation['time'].values



    def annotation_intervals(self):

        '''
        I need this function to return intervals of the annotated notes...

        '''


        pass



    def onset_strength(self, envelope_type ='mean', high_pass=True):


        if high_pass:
            y = highpass_filter(self.y, self.sr,filter_stop_freq=4000, filter_pass_freq=6000)
        else:
            y = self.y
        sr = self.sr

        if envelope_type == 'mean':

            return  librosa.onset.onset_strength(y=y, sr=sr)
        
        if envelope_type == 'median':
        
            return librosa.onset.onset_strength(y=y, sr=sr,aggregate=np.median, fmax=8000, n_mels=256)
        
        if envelope_type == 'cqt':
            return librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.cqt)

        return 'incorrect envelope type'




    def notes_to_hz(self, note):
        '''

        This function will return the freuqncy in hz from a note using librosa. 
        If 
        the note is out of range, then the note will be defaulted to low E or someting...
        '''
        if note == 'X': 
            return 440

        else: 
            return librosa.note_to_hz(note)




    def generate_estimated_annotation(self):

        '''
        This currently has time frames and pitches for the derived song.  Weill this be enough to plug the output of this into 
        the transcription evaluation mir_eval function?

        The output of this functions is an array with 3 columns and rows for each nots representing the time of onset, the time off
        offset and the frequency was mapped to.  

        Wehn I consider the training examples I have prolems with the pitches not being rounded..But maybe I will have to do that myself.


        '''
        ## We have the times of the onsets and off sets of each note.. Now we must derive the frequency,,
        onsets_offsets_pairs, notes = self.detect_note_offsets()
        hz_list = [notes_to_hz(note) for note in notes]
        
        onsets_offsets_array = np.array(onsets_offsets_pairs)

        note_tabulation_array =  np.zeros((len(onsets_offsets_array),3))


        note_tabulation_array[:,0] = librosa.frames_to_time(onsets_offsets_array[:,0])
        note_tabulation_array[:,1] = librosa.frames_to_time(onsets_offsets_array[:,1])
        note_tabulation_array[:,2] = hz_list

        #return onsets_offsets_pairs, notes,onsets_offsets_array

        return note_tabulation_array



    def generate_reference_annotation(self):


        '''
        This method will index into the 

        '''







    def plot_onset_strength(self, onset_envelope = 'standard', plot_all = False, high_pass = True):


        if high_pass:
            y = highpass_filter(self.y, self.sr,filter_stop_freq=25, filter_pass_freq=4000)
        else:
            y = self.y
        sr = self.sr

        D = np.abs(librosa.stft(y))
        times = librosa.frames_to_time(np.arange(D.shape[1]))

        onset_env = self.onset_strength(envelope_type= 'mean', high_pass=True)

        plt.subplot(3,1,1) 
        plt.plot(times, onset_env / onset_env.max(), alpha=0.8, label='Mean (mel)')

        plt.subplot(3, 1,2)

        onset_env = self.onset_strength(envelope_type='median')
        plt.plot(times, onset_env / onset_env.max(), alpha=0.8,label='Median (custom mel)')


        plt.subplot(3,1,3)

        onset_env = self.onset_strength(envelope_type = 'cqt')
        plt.plot(times, onset_env / onset_env.max(), alpha=0.8,label='Mean (CQT)')
    
        #plt.plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,label='Median (custom mel)')
        #plt.plot(times, onset_env / onset_env.max(), alpha=0.8,label='Mean (CQT)')
     #   plt.legend(frameon = True, framealpha = .75)
      #  plt.ylabel('Normalized strength')
        plt.axis('tight')
        plt.tight_layout()

        plt.show()



    def onset_detection(self,wait=0, pre_avg=1, post_avg=20, pre_max=1, post_max=10, delta=.1, high_pass = True, envelope_type = 'mean'):

        if high_pass:
            y = highpass_filter(self.y, self.sr, filter_stop_freq=4000, filter_pass_freq= 6000)
        else:
            y = self.y
        
        D = np.abs(librosa.stft(y))
        sr =  self.sr

        onset_env = self.onset_strength(envelope_type=envelope_type, high_pass=high_pass)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, 
                                          sr=self.sr, 
                                          wait=wait, 
                                          pre_avg=pre_avg, 
                                          post_avg=post_avg, 
                                          pre_max=pre_max, 
                                          post_max=post_max,
                                          delta = delta  )

        
        return onset_env, times, onset_frames
        
  
    def plot_onset(self, plot_onset_env = True, plot_annotations_onsets= False, plot_beats = False, high_pass = False, delta = .5, wait = 20):

      
        y = self.y
            
        D = np.abs(librosa.stft(y))
        
        onset_env, times, onset_frames = self.onset_detection(delta = delta, high_pass=high_pass)

        annotation_offset_frames = 0

        tempo, beats = self.tempo_and_beats()


        plt.figure(figsize= (15,12))
        plt.plot(times[:500], onset_env[:500], label='Onset strength')
        
        plt.vlines(times[onset_frames[onset_frames < 500]] , 0, onset_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
        
        ### plot the annotation note onsets

        if plot_annotations_onsets:

            annotation_onset_frames = librosa.time_to_frames(self.annotation_onsets())
            
            plt.vlines(times[annotation_onset_frames[annotation_onset_frames < 500]],0, onset_env.max(), alpha = .7,color = 'g', linestyle = '--', label = 'Annotation onsets')
        
        if plot_beats:
            plt.vlines(times[beats], 0, onset_env.max(), color = 'c', alpha =.7, linestyle = '--', label = 'Beats')
        
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)

        plt.show()



    def onset_eval(self):

        '''

        This method will use mir_eval to evaluate the onset detection. 

        Onsets should be provided in the form of 1 dimensional array of onsets in seconds in increasing orde

        mir_eval.onset.f_measure(): Precision, REcall, and F-measure scores based on the number of estimated onsets
        which are sufficiently close to reference onsets.

                mir_eval.onset.validate(reference_onsets, estimated_onsets)


                mir_eval.onset.f_measure(reference_onsets, estimated_onsets, window=0.05)


        '''

        onset_env, times, onset_frames = self.onset_detection(high_pass=True)
        estimated_onsets = librosa.frames_to_time(onset_frames, self.sr)
        reference_onsets = self.annotation_onsets()
        F_Measure , Precision, Recall = mir_eval.onset.f_measure(reference_onsets, estimated_onsets)    
        return (Precision,Recall)



    def transcription_eval(self):

        '''

        This method will attempt to evaluate the transcription that I have generated using mir_eval.    

        '''
        onset_offset_pairs, notes = self.detect_note_offsets()



        return (onset_offset_pairs, notes)


    def extract_notes(self, high_pass = False):

        notes = []

        if high_pass:
            y = highpass_filter(self.y, self.sr)
            D = np.abs(librosa.stft(y))
        else:
            y = self.y
            D = np.abs(librosa.stft(y))

        cqt = librosa.core.cqt(y, sr=self.sr)
        onset_env, onset_times, onset_frames = self.onset_detection(delta = .1, high_pass=True, envelope_type='cqt')

        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr)

        note_indices = np.argmax(abs(cqt[:,[onset_frames]]), axis = 0)

        for note in note_indices[0]:

            if note+1 > 60:
                notes.append('X')
            else:
                notes.append(cqt_index_to_note[note+1])

        return notes, onset_frames


    def cqt(self):

        return librosa.core.cqt(self.y,sr = self.sr)



    def onset_intervals(self,units = 'time'):

        if units == 'time':

            return librosa.frames_to_time(self.detect_note_offsets())

    
        return self.detect_note_offsets()


    def detect_note_offsets(self):
        '''
        This method will attempt to determine the length of a note.

        The naive method of doing this would be to see how long it takes
        for the note to lose a certain fraction of its amplitutde at that 
        given frequency in the spectrum....  

        I want this method to return a tuple.  Of the onsets time and and offset times,
        '''

        ## Initalize an arry of onset frames, and off set frames

        notes, onset_frames = self.extract_notes()

        cqt = self.cqt()
        abs_cqt = np.abs(cqt)

        note_indices = np.argmax(abs_cqt[:,[onset_frames]], axis = 0)

        offset_frames = np.zeros(onset_frames.shape[0])

        onset_offset_pairs = []


        for i in range(len(note_indices[0])):

            ## Index into the CQT spectrum.  

            onset = onset_frames[i]
            note = note_indices[0][i]
            timeseries_at_freq = abs_cqt[note,onset : onset +20]     

            offset = self.attenuated_timeseris(onset, .5, timeseries_at_freq)
            onset_offset_pairs.append((onset, offset))
            #return timeseries_at_freq
            #print("The shape of the timeseries is " + str(timeseries_at_freq.shape)   )
            #offset = self.attenuated_timeseris(current_onset , .5, timeseries_at_freq)
            #onset_offset_pairs.append(current_onset,offset)

        return onset_offset_pairs, notes


    

    def attenuated_timeseris(self,  onset, attenuation_factor, timeseries):

        ''' 
        This method will determine the the time frame at which the time seres has attenuated.. 
        '''

        peak_ampltiude = timeseries[0]

        offset =onset

        for k in range(timeseries.shape[0]):

            if timeseries[k]  < attenuation_factor * peak_ampltiude:
                return offset + k 
                
        ## By default what do I what this ....
        return offset+20


    def plot_note_offset(self):

        '''
        This method will be used primarily to generate of the offset detecion funcion..

        '''
        pass


    def tempo_and_beats(self):
        
        return  librosa.beat.beat_track(y = self.y, sr = self.sr)  



    def estimate_pitch(self, segment, sr, fmin=50.0, fmax=2000.0):
        
        # Compute autocorrelation of input segment.
        r = librosa.autocorrelate(segment)
        
        # Define lower and upper limits for the autocorrelation argmax.
        i_min = sr/fmax
        i_max = sr/fmin
        r[:int(i_min)] = 0
        r[int(i_max):] = 0
        
        # Find the location of the maximum autocorrelation.
        i = r.argmax()
        f0 = float(sr)/i
        return f0


    def plot_spectrum(self):

        librosa.display.specshow(self.S,y_axis = 'log')
        plt.colorbar(format ='%+2.0f dB')
        plt.title('Log spectrum')
        plt.show()

    def plot_log_power(self):

        librosa.display.specshow(self.D,y_axis = 'log')
        plt.colorbar(format ='%+2.0f dB')
        plt.title('Log-frequency power spectrum')
        plt.show()



    def plot_cqt_note(self):


        CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(self.y, sr = self.sr)), ref = np.max)
        librosa.display.specshow(CQT, y_axis = 'cqt_note')
        plt.colorbar(format ='%+2.0f dB')
        plt.title('Constant-Q power of spectogram (note)')
        plt.show()


    def plot_cqt_hz(self):
        CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(self.y, sr = self.sr)), ref = np.max)
        librosa.display.specshow(CQT, note ='cqt_hz')
        plt.colorbar(format ='%+2.0f dB')
        plt.title('Constant-Q power of spectogram (HZ)')


    def draw_chromagram(self):

        C = librosa.feature.chroma_cqt(y = self.y, sr = self.sr)
        librosa.display.specshow(C, y_axis = 'chroma')
        plt.colorbar
        plt.title('Chromagram')
        plt.show()



def extract_tempo_and_beats(data, fs):


    ''' 
    This function will extract beats and tempo from the audio file by calling librosa

    '''
    bpm,beats = librosa.beat.beat_track(data, fs)
    beat_times = librosa.frames_to_time(beats, sr=fs)    

    return bpm, beat_times





def time_beat_indices(t, beats):

    '''
    This function will return the indices of t which are closest to the times at which beats occur...

    '''

    indices = np.zeros(beats.shape[0])
    for i in range(beats.shape[0]):
        indices[i] = np.argmin(np.abs(t-beats[i]))
        
    return indices



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



def jam_to_dataframes(jam_file):

    ''' 
    This function will return a dictionary of dataframes for the given jam_file...
    '''

    annotation = (jams.load(jam_file)).annotations

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

    frames = []
    for dataframe in jams_dictionary:

        jams_dictionary[dataframe].drop('confidence', axis =1 , inplace = True)

        jams_dictionary[dataframe]['string'] = dataframe
        frames.append(jams_dictionary[dataframe])

    strings_df = pd.concat(frames)
    strings_df.sort_values('time', inplace=True)

    return strings_df



cqt_index_to_note = {

    1:'C1',
    2:'C#1',
    3:'D1',
    4:'D#1',
    5:'E1',
    6:'F1',
    7:'F#1',
    8:'G1',
    9:'G#1',
    10:'A1',
    11:'A#1',
    12:'B1',
 
    13:'C2',
    14:'C#2',
    15:'D2',
    16:'D#2',
    17:'E2',
    18:'F2',
    19:'F#2',
    20:'G2',
    21:'G#2',
    22:'A3',
    23:'A#3',
    24:'B3',

    25:'C3',
    26:'C#3',
    27:'D3',
    28:'D#3',
    29:'E3',
    30:'F3',
    31:'F#3',
    32:'G3',
    33:'G#3',
    34:'A4',
    35:'A#4',
    36:'B4',

    37:'C4',
    38:'C#4',
    39:'D4',
    40:'D#4',
    41:'E4',
    42:'F4',
    43:'F#4',
    44:'G4',
    45:'G#4',
    46:'A5',
    47:'A#5',
    48:'B5',

    49:'C5',
    50:'C#5',
    51:'D5',
    52:'D#5',
    53:'E5',
    54:'F5',
    55:'F#5',
    56:'G5',
    57:'G#5',
    58:'A6',
    59:'A#6',
    60:'B6',

}



def highpass_filter(y, sr, filter_stop_freq = 60, filter_pass_freq = 100):

    filter_order = 1001
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

    # Apply high-pass filter
    filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio



def notes_to_hz(note):
    if note == 'X':
        return 440
    else: 
        return librosa.note_to_hz(note)
