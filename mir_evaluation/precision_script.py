'''

In this script I will ne

'''


import numpy as np


from tabgen.spectogram import Spectogram

def transcription_onset_recall_evaluation(n):


    '''
    This function will calculate the precision and the recall for the data in the training set. 

    '''

    transcription_precision_recall = np.zeros((n, 2))


    counter = 0 
    for spec_id in range(n):

        spec = Spectogram(spec_id)
        #transcription_precision_recall[i 0], transcription_precision_recall[i, 1] = spec.transcription_evaluation()
       

        transcription_precision_recall[spec_id,0] , transcription_precision_recall[spec_id,1] = spec.transcription_evaluation()

        counter+= 1
        print(counter)

    return transcription_precision_recall


def onset_recall_evaluation(n):

    '''
    This function will calculate the precision and recall forr each of the Spectograms in the training directory...

    '''
    onsets_precison_recall = np.zeros((n,2))
    counter = 0 
    for i in range(n):

        spec = Spectogram(i)
        PrecisionRecall = spec.onset_eval()
        #P,R = spec.onset_eval()

        onsets_precison_recall[i, 0] = PrecisionRecall[0]
        onsets_precison_recall[i,1] = PrecisionRecall[1]
        counter += 1
        print(counter)
    return onsets_precison_recall


