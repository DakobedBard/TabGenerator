'''

This file will have functions used for recording and storing audio using the built in sound API.  

'''

import pyaudio
import wave
from scipy.io import wavfile
import os
#from tabgen.tablature import Note, Tab
#from tabgen.spectogram import Spectogram

import pyaudio
import wave
import sys


def play_audio(wave_file_path):
    CHUNK = 1024
    

    p = pyaudio.PyAudio()
    try:
        
        wf = wave.open(wave_file_path, 'rb')

        # instantiate PyAudio (1)
        

        # open stream (2)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # read data
        data = wf.readframes(CHUNK)

        # play stream (3)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(CHUNK)

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        # close PyAudio (5)
        p.terminate()
    except KeyboardInterrupt:
        p.terminate()


def recording_file_name(get_raw_input = True):

    if get_raw_input:

        name = input("filename: ")
    return name



def load_wavfile(wav_file):

    '''
    This function will be a wrapper for the scipy.io's wavfile read function

    '''

    fs, data = wavfile.read(wav_file)

    return fs , data


def record(filename):

    '''
    This function will record audio until the user clicks control C.  This function will have to be modified slightly in order
    for other interupts to be used... Check that the specified file name has not be been already used.

    I think that at some point I will want to 

    '''
    exists = os.path.isfile(filename)

    if exists:

        raise ValueError('A file with this name already exists... ')
        

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    
    WAVE_OUTPUT_FILENAME = filename
    
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
    
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

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
        



def record_audio(filename):

    '''

    This function will record an audio recording and save it at the location of filename. 


    If a file already exists with the same name, I will have to decide what to do... Most likely promt the user to choose a different filename.  

        - Ask how this would be handled.... 

        - First I have to check wether the file exists or not...


    Currently the function will only record for 5 second so I will need to add some functionality to receive user input.  


    '''


    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = filename
    
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
    
    
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()


