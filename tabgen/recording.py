'''
This file wll contains class declarations for the Recording and RecordingCollection class.  

'''

import pyaudio
import wave
from scipy.io import wavfile
import os
from tabgen.tablature import  Tab
from tabgen.spectogram import Spectogram, annotation_audio_file_paths

import pyaudio
import wave
import sys




class RecordingCollection():
    '''
    This class will represent a collection of recordings.  

    When the class is initialized I would like to add all the files in the directory to a list 

    '''

    def __init__(self,recordings_directory):

        if not os.path.isdir(recordings_directory): 

            os.mkdir(recordings_directory)
        
        self.recordings_directory = recordings_directory
    
        self.wav_files = os.listdir(self.recordings_directory)
        self.wav_file_paths = [self.recordings_directory+wav_file for wav_file in self.wav_files ]



    def create_recording(self):


        new_recording = Recording(self.recordings_directory)
        try:
        
            new_recording.record()
        except OSError:
            pass
        self.wav_files.append(new_recording.fname.split('/')[-1])
        self.wav_file_paths.append




    def genereate_chroma(self, wav_file_name):

        if os.path.isfile(wav_file_name):  
            rec1 = Recording(filepath = wav_file_name)
            rec1.spec.plot_chroma_cqt(plot=True)  


    def generate_onsets(self, wav_file_name):
            
            if os.path.isfile(wav_file_name):            
                print("There is a wav file here ")

                rec1 = Recording(filepath = wav_file_name)
               # onset_env, times, onset_frames, onset_boundaries, onset_times, beats= rec1.spec.onset_detection(plot=True) 

                #return  onset_frames, onset_boundaries, onset_times, beats



    def generate_spectogram(self, wav_file_name):
        if os.path.isfile(wav_file_name):            
            print("There is a wav file here ")

            rec1 = Recording(filepath = wav_file_name)
           # rec1.spec.plot_onsets(plot=True)  

            #rec1 = Recording(self.recordings_directory, wav_file_name)
            #return rec1

    def play_audio(self, wav_file_name):


        '''
        This function will play audio when called.  

        '''

class Recording():


    '''

    This class represents an individual recording.  A recording should be able to be initialized in one of two ways.  A new recording and a recording
    for which the file path to the wav file is provided.  


    '''

    def __init__(self,recording_directory= None,filepath = None):


        self.recording_directory = recording_directory
        self.wav_file_path = filepath
        
        print(self.wav_file_path)


        if filepath == None:
            pass
        else:    
            self.spec = Spectogram(self.wav_file_path)


    def plot_spectogram_onsets(self):

        #self.spec.plot_onsets(plot=True)
        pass

    def record(self):

        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024    
        audio = pyaudio.PyAudio()

        frames = []
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

        print('Recording')
        while True:

            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except KeyboardInterrupt:
                # stop Recording
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
                print('Finished Recording')
                name = input('Filename: ')
                
                validname = False
                while name == False:
                    
                    #validname = not os.isfile(self.recording_directory+name+".wav")
                    if os.path.isfile(self.recording_directory+name+".wav"):
                        name = input('a file with that name exists.  Please enter another filename')
                    validname = not os.path.isfile(self.recording_directory+name+".wav")

                self.wav_file_path = self.recording_directory + name +'.wav'
                WAVE_OUTPUT_FILENAME = self.wav_file_path
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                self.spec = Spectogram(self.wav_file_path)
                self.fname = WAVE_OUTPUT_FILENAME

def recording_file_name(get_raw_input = True):

    if get_raw_input:

        name = input("filename: ")
    return name


