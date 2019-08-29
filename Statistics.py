"""
Compute statistics of reasoning and memory 
on modified WCST version from Dehaene and Changeux
implemented model

Pauline Bock - Mnemosyne Team (INRIA)
22/05/2019
"""

from DehaeneChangeux import WCST_test
import numpy as np 

speed_lr_total = []
single_trial_lr_total = []
perseveration_total = []
nbTS_total = []


for t in range(0, 500):
    print("trial " + str(t))
    speed_lr, single_trial_lr, perseveration, nbTS = WCST_test()
    speed_lr_total.append(speed_lr)
    single_trial_lr_total.append(single_trial_lr)
    perseveration_total.append(perseveration)
    nbTS_total.append(nbTS)

#MEAN on 500 trials
speed_lr_mean = np.mean(speed_lr_total)
single_trial_lr_mean = np.mean(single_trial_lr_total)
perseveration_mean = np.mean(perseveration_total)
nbTS_mean = np.mean(nbTS_total)

print("STATISTICS on 500 trials:")
print("Speed of learning: " + str(speed_lr_mean))
print("Single-trial learning: " + str(single_trial_lr_mean))
print("Perseverations: " + str(perseveration_mean))
print("nbTS mean: " + str(nbTS_mean))