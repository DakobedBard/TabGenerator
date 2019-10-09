import jams
import pandas as pd
import os
import librosa
import librosa.display

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from tabgen.spectogram import Spectogram

'''
This file will define a Note, Measure and a Tab class.  

'''

E = ['E2','F2','F#2','G2','G#2','A3','A#3','B3','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3']
A = ['A3','A#3','B3','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A4','A#4','B4','C4','C#4','D4']
D = ['D3','D#3','E3','F3','F#3','G3','G#3','A4','A#4','B4','C4','C#4','D4','D#4','E4','F4', 'F#4','G4']
G = ['G3','G#3','A4','A#4','B4','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A5','A#5', 'B5','B#5']
B = ['B4','C4','C#4','D4','D#4','E4','F4','F#4','G4', 'G#4','A5','A#5','B5','B#5','C5','C#5','D5','D#5']
hE= ['E4','F4', 'F#4','G4', 'G#4','A5','A#5','B5','B#5','C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5']

standard_tuning =  [E,A,D,G,B,hE]

lowD = E = ['D2','D#2','E2','F2','F#2','G2','G#2','A3','A#3','B3','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3',]

drop_D_tuning = [ lowD, A, D,G,B,hE]
    



class Measure():


    '''
    This class will represent a measure that will be sent to the flask application to be displayed.
    A measure will be initialized from a list of notes.  

    '''

    def __init__(self, notes):

        self.notes = notes
        self.genereate_measure_html()

    def generate_note(self, note):

        '''
        This method will find which string to play the note on... 
        '''

        string_fret_combinations = []

        for string in range(len(standard_tuning)):
            for fret in range(len(standard_tuning[0])):
                if standard_tuning[string][fret] == note:
                    string_fret_combinations.append((string, fret))
         

        return string_fret_combinations


    def generate_note_strings(self, note):

        if note =='Out of range...':
            return 'xxxxxx', ''

        string_fret_combinations = self.generate_note(note)
        try:


            note_string = string_fret_combinations[-1][0]
            note_fret = string_fret_combinations[1][1]
            return note_string, note_fret
        except:
            return 0, 6


    def genereate_measure_html(self):

        #tab = [[s+'|'] for s in ['E', 'A','D','G','B','e']]
        tab = [[] for s in range(6)]

        for note in self.notes:

            (string, fret) = self.generate_note_strings(note)

            for string_id in range(6):
                if string_id == string:
                    tab[string_id].append(str(fret).rjust(2, '-'))
                else:
                    tab[string_id].append('--')


        for string_id in range(6):
            tab[string_id].append('|')

        tab= tab[::-1]
        tab_text = '\n'.join('-'.join(tab_string) for tab_string in tab)

        self.measure_text = tab_text



class Line():

    '''

    This seems as though it may be a necessary abstraction ... 

    I will have to rework the code so that I no longer have the measure abstraction.. or the generate

    generate_measure_html method becomes generae_line_html


    '''

    def __init__(self, measures_per_line):
        

        self.measures_per_line = measures_per_line
        self.measures = []
        

    def add_measure(self, measure):
        '''
        This method will be be used to 

        '''

      # measure_split = [[measure.split('\n')] for measure in measures] 
        #print("I get here ")
        self.measures.append(measure)
        
    def generate_line_html(self):

        tab = [[s+'|'] for s in ['E', 'A','D','G','B','e']]
        

        print("THere are " + str(len(self.measures)) + " measures in this line")
        for measure in self.measures:
            #print("I get here ")
            for note in measure.notes:
                (string, fret) = measure.generate_note_strings(note)
                
                for string_id in range(6):
                    if string_id == string:
                        tab[string_id].append(str(fret).rjust(2, '-'))
                    else:
                        tab[string_id].append('--')

            for string_id in range(6):
                tab[string_id].append('|')

        tab= tab[::-1]
        tab_text = '\n'.join('-'.join(tab_string) for tab_string in tab)
        tab_text += '\n'
        self.line_html = tab_text


    def print_line(self):

        '''
        This method will print out the line  
        '''

        print(self.line_html)


class Tab():

    '''
    This class will represent a guitar tab.  

    It would appear as though in the process of writing this code the need for a Line class as well as Measure class
    appeared necessary... Is it necesarry...?


    '''
    def __init__(self, wav_file_name):

        self.spec = Spectogram(wav_file_name)
    
        self.notes, self.onset_frames = self.spec.extract_notes()

        self.measures_per_line =4

        self.split_into_measures()
        self.generate_lines()
        

    def split_into_measures(self):

        '''
        This method will attempt to split the notes up into measures,  possibly based on the number of beats

        '''
        tempo, beats = self.spec.tempo_and_beats()

        onset_frames = self.onset_frames
        measure_frames = beats[::4]
        n_measures = measure_frames.shape[0]
        measures = []
        note_count = 0

        for measure_count in range(len(measure_frames)):
            notes = []
            while onset_frames[note_count] < measure_frames[measure_count]:
                notes.append(self.notes[note_count])
                note_count += 1

            m = Measure(notes)
            measures.append(m)

        self.measures = measures






    def generate_lines(self):
        '''
        This method is used to generate lines of measures. 

        A line of tablature is a list of measures 
        
        
        I am using a custom Lines class to represent the list of measures...
        Therefore this method will return a list of Lines objecs... .     


        '''

        measures = self.measures 
        n_measures = len(measures)

        nlines = n_measures//4

        if n_measures%4 >0:
            nlines += 1

        lines = []
        ## This is ugly code I know!

        for i in range(nlines):

            current_line = Line(self.measures_per_line)

            current_line.add_measure(measures[4*i])
            if 4*i+1 > n_measures - 1:

                current_line.generate_line_html()
                lines.append(current_line)
                break
            current_line.add_measure(measures[4*i+1])
            if 4*i+2 > n_measures - 1:
                current_line.generate_line_html()
                lines.append(current_line)
                break
            current_line.add_measure(measures[4*i+2])
            if 4*i+3 > n_measures - 1:
                current_line.generate_line_html()
                lines.append(current_line)
                break
            
            current_line.add_measure(measures[4*i+3])
            current_line.generate_line_html()
            lines.append(current_line)
        

        self.lines = lines
        self.nmeasures = n_measures
    
        ## For each line ... call the concatenate_measures


    def print_tab(self):

        '''

        This method shoudld print out the entire tab. 

        I should probably have some method at some point that will work with the web application


        '''

        for line in self.lines:
            line.print_line()

            

                




