"""
Compute statistics of reasoning and memory 
on modified WCST version from Dehaene and Changeux
implemented model

Pauline Bock - Mnemosyne Team (INRIA)
22/05/2019
"""

from DehaeneChangeux import WCST_test
import numpy as np 
import time

speed_lr_total = []
single_trial_lr_total = []
perseveration_total = []
nbTS_total = []
time_total = []
trials_total = []

start_stat = time.time()

for t in range(0, 500):
    print("trial " + str(t))
    nb_trials, speed_lr, single_trial_lr, perseveration, nbTS, test_time = WCST_test()
    speed_lr_total.append(speed_lr)
    single_trial_lr_total.append(single_trial_lr)
    perseveration_total.append(perseveration)
    nbTS_total.append(nbTS)
    time_total.append(test_time)
    trials_total.append(nb_trials)

end_stat = time.time()

#MEAN on 500 trials
speed_lr_mean = np.mean(speed_lr_total)
single_trial_lr_mean =  np.mean(single_trial_lr_total)
perseveration_mean = np.mean(perseveration_total)
nbTS_mean = np.mean(nbTS_total)
time_mean = np.mean(time_total)
time_stat = end_stat - start_stat
trials_mean = np.mean(trials_total)

print("STATISTICS on 500 trials:")
print("Speed of learning: " + str(speed_lr_mean))
print("Single-trial learning: " + str(single_trial_lr_mean))
print("Perseverations: " + str(perseveration_mean))
print("nbTS mean: " + str(nbTS_mean))
print("nb trials mean: " + str(trials_mean))
print("Mean time: " + str(time_mean))
print("Stat time: " + str(time_stat))

f = open("LesionedReward2.txt", "w")
text = "Speed: " + str(speed_lr_mean) + "\nSingle-trial learning: " + str(single_trial_lr_mean) + "\nperseveration: " + str(perseveration_mean) + "\nnbTS: " + str(nbTS_mean) + "\nnb trials: " + str(trials_mean)
f.write(text)
f.close()