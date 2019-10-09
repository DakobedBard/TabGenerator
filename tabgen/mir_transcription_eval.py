import mir_eval

from tabgen.spectogram import Spectogram
'''

This script will perform evaluation of the various metrics from mir_eval.




'''




def training_onset_evaluation():


    '''

    Do I want to only consider the annotated audio files that are of the solo variety.  I'm note sure how I 
    am going to handle the chords ... 


    '''




    for i in range(360):


        spec = Spectogram(i)


        F,P,R = spec.onset_eval()
        print(i)




