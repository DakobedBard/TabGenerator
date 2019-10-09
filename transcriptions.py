'''
This script will demonstrate how to measure the Recall of the onset detection system for a single training file.  


This script will use mlflow to log the results of the different parameter selections.  Over the entire dataset.
I will have to refactor the code in order to do this as of right now mlflow is only recording th parameters and the metrics for a single


'''

from tabgen.transcriber import Transcriber
from tabgen.jams_io import annotation_audio_file_paths


files = annotation_audio_file_paths()
transciber = Transcriber(files[70][0], files[70][1])

#transciber.evaluate_onset_detection()

transciber.evaluate_onset_detection()