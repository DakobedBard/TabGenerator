
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa.display
import mir_eval


class OnsetDetector:
    '''An instance of this class will be a member of the transcriber class. 
    THe transcriber class will call upon this object to return the estimated onsets.

    '''

    def __init__(self, y, sr=22500, annotation=None, wait=0, pre_avg=1, post_avg=20, pre_max=1, post_max=10, delta=.1,  envelope_type = 'mean'):

        self.wait = wait
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.delta = delta
        self.envelope_type = envelope_type

        self.y = y 
        self.sr = sr

        if annotation is None:
            self.annotation = pd.DataFrame(['time'])
        else:
            self.annotation = annotation
    

    def multi_onset_strength(self):

        '''
        This function will make use of the librosa.onset_strengt_multi function.
    
        '''
        y = self.y
        sr = self.sr
        D = np.abs(librosa.stft(y))
        plt.figure()
        plt.subplot(2,1,1)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref = np.max))
        plt.title('Power Spectogram')
        
    
        onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,channels=[0, 32, 64, 96, 128])
        plt.subplot(2, 1, 2)
        librosa.display.specshow(onset_subbands, x_axis='time')
        
        #plt.show()
        return onset_subbands  

    def multi_onset_detect(self):

        pass

    def onset_strength_envelope(self, envelope ='mean'):
        '''
        This function will return an onset envelope
        '''
        y = self.y
        sr = self.sr


    def onset_strength(self, envelope_type='mean'):

        y = self.y
        sr = self.sr

        if envelope_type == 'mean':

            return  librosa.onset.onset_strength(y=y, sr=self.sr)
        
        if envelope_type == 'median':
        
            return librosa.onset.onset_strength(y=y, sr=sr,aggregate=np.median, fmax=8000, n_mels=256)
        
        if envelope_type == 'cqt':
            return librosa.onset.onset_strength(y=y, sr = sr, feature=librosa.cqt)

        return 'incorrect envelope type'


    def onset_detection(self):
    
            D = np.abs(librosa.stft(self.y))
            sr =  self.sr
            onset_env = self.onset_strength(envelope_type=self.envelope_type)
            times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
            
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, 
                                            sr=self.sr, 
                                            wait=self.wait, 
                                            pre_avg=self.pre_avg, 
                                            post_avg=self.post_avg, 
                                            pre_max=self.pre_max, 
                                            post_max=self.post_max,
                                            delta = self.delta  )

            
            return onset_env, times, onset_frames


    def plot_onset_strength(self, onset_envelope = 'standard'):

            D = np.abs(librosa.stft(self.y))
            times = librosa.frames_to_time(np.arange(D.shape[1]))

            onset_env = self.onset_strength(envelope_type= 'mean')

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


    def plot_onset(self, plot_onset_env = True, plot_annotations_onsets= False, plot_beats = False):


        y = self.y            
        D = np.abs(librosa.stft(y))
        
        onset_env, times, onset_frames = self.onset_detection()
        annotation_offset_frames = 0

        plt.figure(figsize= (15,12))
        plt.plot(times[:500], onset_env[:500], label='Onset strength')
        
        plt.vlines(times[onset_frames[onset_frames < 500]] , 0, onset_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
        
        ### plot the annotation note onsets

        if plot_annotations_onsets:

            annotation_onset_frames = librosa.time_to_frames(self.annotation_onsets())
            
            plt.vlines(times[annotation_onset_frames[annotation_onset_frames < 500]],0, onset_env.max(), alpha = .7,color = 'g', linestyle = '--', label = 'Annotation onsets')
        
        # if plot_beats:
        #     plt.vlines(times[beats], 0, onset_env.max(), color = 'c', alpha =.7, linestyle = '--', label = 'Beats')
        
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)

        plt.show()

    def annotation_onsets(self, units = 'time'):
        
        if units == 'frames':
        
            return librosa.time_to_frames(times = self.annotation['time'].values, sr = self.sr)

        if units == 'time':
            return self.annotation['time'].values

    def onset_eval(self):

        '''

        This method will use mir_eval to evaluate the onset detection. 

        Onsets should be provided in the form of 1 dimensional array of onsets in seconds in increasing orde

        mir_eval.onset.f_measure(): Precision, REcall, and F-measure scores based on the number of estimated onsets
        which are sufficiently close to reference onsets.

                mir_eval.onset.validate(reference_onsets, estimated_onsets)


                mir_eval.onset.f_measure(reference_onsets, estimated_onsets, window=0.05)


        '''

        onset_env, times, onset_frames = self.onset_detection()
        estimated_onsets = librosa.frames_to_time(onset_frames, self.sr)
        reference_onsets = self.annotation_onsets()
        F_Measure , Precision, Recall = mir_eval.onset.f_measure(reference_onsets, estimated_onsets)    
        return (Precision,Recall)