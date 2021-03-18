### This is a dummy code for sanity check, printing out numbers/queries/labels for us to easily see! 

import numpy as np
from enum import Enum
from collections import Counter 
from time import time
import cv2
import csv
import os

from LowLevelFigure import LowLevelFigure
from MidLevelFigure import MidLevelFigure
from HighLevelFigure import HighLevelFigure

# The type of data classes we will be generating
class Dataclass(Enum):
    LENGTH = 0
    LENGTHS = 1
    ANGLE = 2
    ANGLES = 3
    SIMPLE_BAR=4 # simple bar plot, level 2
    SIMPLE_PIE=5 # simple pie plot, level 2
    ADVANCED_BAR=6 # advanced bar plots, level 3
    ADVANCED_PIE=7 # advanced pie plots, level 4

### CHANGE THIS PART FOR DATASET WITH DIFFERENT SIZE
path = './sanity_check_fig/'

# Reads in csv
metadata_table = []

total_files = os.listdir(path)
## Reads metadata
for filename in total_files:
    if filename[-3:] == 'csv':
        with open(path+filename, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader: 
                if row[0] in total_files: 
                    metadata_table = metadata_table + [row]
    elif filename[-3:] == 'png':
        pass
        #img = cv2.imread(path+filename)
        #im_table = im_table + [img]
    else: 
        assert False, "Wrong file format"

#assert len(im_table) == len(metadata_table), "Did not find the right number of metadata"


for filename in os.listdir(path):
    if filename[-3:] == 'png': 
        # Reads in the image
        im = cv2.imread(path+filename)
        # Finds corresponding metadata
        for md in metadata_table:
            if md[0] == filename:
                im = np.vstack((im, np.ones(im.shape)*200))
                font = cv2.FONT_HERSHEY_SIMPLEX
                TEXTSCALE = 0.5
                cv2.putText(im, md[0], (20, 420), font, TEXTSCALE, 0)
                cv2.putText(im, md[1], (20, 470), font, TEXTSCALE, 0)
                cv2.putText(im, md[2], (20, 520), font, TEXTSCALE, 0)
                cv2.putText(im, md[3], (20, 570), font, 0.35, 0)
                cv2.putText(im, md[4], (20, 620), font, 0.35, 0)
                # Edits the image and exports
                cv2.imwrite("Annotated_"+str(filename), im)
                break


