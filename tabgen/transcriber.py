from tabgen.onset_detection import OnsetDetector
import librosa

from tabgen.jams_io import tablature_dataframe

import mlflow




import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow.sklearn




class Transcriber:
    ''' This class will have an instance of the Onset Detector, Pitch Identifier and 
    '''

    def __init__(self, filename, annotation_filename=None):

        self.filename = filename
        self.name = filename.split('/')[-1].split('.')[0]
        self.y, self.sr = librosa.load(filename)

        if annotation_filename  is None:
            self.annotation = None
        else:
            self.annotation = tablature_dataframe(annotation_filename)

        self.onsetdetector = OnsetDetector(self.y, self.sr, self.annotation)
        
    def plot_onset_strength(self):
        self.onsetdetector.plot_onset_strength()

    def return_onset_detector(self):
        return self.onsetdetector



    def log_parameters(self):


        mlflow.create_experiment('linear_bullshit', '/home/mddarr/galvanize/capstone/tab_generator/mlflow_artifacts')


        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        print("Score: %s" % score)
        mlflow.log_metric("score", score)
        mlflow.sklearn.log_model(lr, "model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid) 



    def evaluate_onset_detection(self):
        '''
        This method will... Create an onsetdetector object, and will initialize it with a range of values of parameters..
        The onsetdetector will need to have a method that will allow me to change the its parameters.  

        '''
        ## 
        wait=0 
        pre_avg=1
        post_avg=20 
        pre_max=1 
        post_max=10
        delta=.1 
        envelope_type = 'mean'

        # wait = wait
        # pre_avg = pre_avg
        # post_avg = post_avg
        # pre_max = pre_max
        # post_max = post_max
        # delta = delta
        # envelope_type = envelope_type



        ## Proably write some methods in the OnseDetectr that will change the values instead of calling the constructor over and over

        self.onsetdetector = OnsetDetector(self.y, self.sr, self.annotation, wait= wait, pre_avg = pre_avg, 
                    post_avg = post_avg,pre_max = pre_max,post_max = post_max
                    ,delta = delta , envelope_type = envelope_type)

        with mlflow.start_run() as run:


            precision, recall = self.onsetdetector.onset_eval()


            mlflow.log_param("wait", wait)
            mlflow.log_param("pre_avg", pre_avg)
            mlflow.log_param("post_avg", post_avg)
            mlflow.log_param("pre_max", pre_max)
            mlflow.log_param("post_max", post_max)
            mlflow.log_param("delta", delta)
            mlflow.log_param("envelope_type", envelope_type)
 
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)

            print(precision)
            print(recall)
            