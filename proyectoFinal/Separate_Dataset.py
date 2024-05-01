"""
#Separate Dataset
import splitfolders
import numpy as np 
import pandas as pd 

import splitfolders
dataset = "C:/Users/kevin/Desktop/ASL_Dataset/asl_alphabet"

splitfolders.ratio(dataset, output="datasets/asl_alphabet",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

train_dir = 'datasets/asl_alphabet/train'
val_dir = 'datasets/asl_alphabet/val'
test_dir  = 'datasets/asl_alphabet/test'
"""


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('C:/Users/kevin/Desktop/ASL_Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        