'''

This script will be used to tune the onset detection algorithm. 


things to do.


modify my onset detection algorithm such that it can be passed in paramters...



Maybe I should ask somebody some questions about performing a grid search in search of the ideal parameters... 




'''





from tabgen.spectogram import Spectogram
spec = Spectogram(70)

wait =0
pre_avg = 1
post_avg = 1
pre_max = 1
post_max = 10
delta = .1
pre_avg  = 0
post_avg = 0

pre_max = 0

def n_onset(spectogram, wait = wait,  pre_avg = pre_avg, post_avg = post_avg, pre_max = pre_max, post_max = post_max, delta = delta):

    onset_env, times, onset_frames, onset_boundaries, onset_times = spectogram.onset_detection(wait = wait, 
                                                                                        pre_avg = pre_avg, 
                                                                                        post_avg = post_avg, 
                                                                                        pre_max = pre_max, 
                                                                                        post_max = post_max,
                                                                                        delta = delta)


