### Code started 2021 Feb
### Sample generation for CQAW competition 
###
### Author: Hayoun Oh
### Thanks to Dr. Yunhai Wang and his group for providing mid-level dataset and generation code
import numpy as np
from enum import Enum
from collections import Counter 
from time import time
import cv2
import csv

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

SIMPLE_QUERIES = ['What is the length of the line in the figure?', 'What are the lengths of the lines in the figure, from left to right?', 
            'What is the size of the angle in the figure?','What are the sizes of the angles in the figure, from left to right?', 
            'What are the values represented in the bar graph, from left to right?', 'What are the values represented in the pie graph, clockwise from the top?']


# Query matchmaker for advanced bar and pie plots
def advanced_query_matcher(dset):
    # dset is a tuple of training dataset and test dataset

    # The following are the queries
    QUERIES = {
        0: ('What is the value of {X}?', (lambda nums, i, j, k: np.around(nums[i],3))), 
        1: ('Which variety has the maximum value?', (lambda nums, i, j, k: np.argmax(nums) if len(np.atleast_1d(np.argmax(nums)))==1 else None)), 
        2: ('Which variety has the minimum value?', (lambda nums, i, j, k: np.argmin(nums) if len(np.atleast_1d(np.argmin(nums)))==1 else None)), 
        3: ('Is {X} bigger than {Y}?', (lambda nums, i, j, k: 'Yes' if nums[i]>nums[j] else ('No' if nums[i]<nums[j] else None))), 
        4: ('Is {X} smaller than {Y}?', (lambda nums, i, j, k: 'Yes' if nums[i]<nums[j] else ('No' if nums[i]>nums[j] else None))), 
        5: ('Is {X} bigger than the sum of {Y} and {Z}?', (lambda nums, i, j, k: 'Yes' if nums[i]>nums[j]+nums[k] else ('No' if nums[i]<nums[j]+nums[k] else None))), 
        6: ('What is the sum of values for {X} and {Y}?', (lambda nums, i, j, k: np.around(nums[i]+nums[j], 3) ))
    }
    n_queries = len(QUERIES)

    queries = []
    labels = []

    # Randomly matches a query with each nums in dset
    for (_, varieties, nums) in dset:
        while True:
            Q_idx = np.random.randint(n_queries)
            formatQuery, funQuery = QUERIES[Q_idx]
            assert len(nums) > 2, str(nums)
            ints = np.random.choice(len(nums),3,replace=False)
            lab = funQuery(nums, ints[0], ints[1], ints[2])
            if lab is not None: 
                strQuery = formatQuery.format(X=varieties[ints[0]], Y=varieties[ints[1]], Z=varieties[ints[2]])
                if Q_idx == 1 or Q_idx == 2: 
                    lab = varieties[lab]
                queries = queries + [strQuery]
                labels = labels + [lab]
                break
            else: 
                print("Invalid query for this figure. Choosing a new query...")

    return queries, labels

# If the full data size is too large, we split it up in batches
def batch_generation(path='./competition_data/', prefix='', batch_idx=0, batch_size=10):
    print("------------------------------------------")
    print("Creating batch "+str(batch_idx))
    data_counts = {'train':int(round(batch_size*0.8)), 'test':batch_size-int(round(batch_size*0.8))}
    print("Composed of "+str(data_counts['train'])+" training data and "+str(data_counts['test'])+" test data")

    train_metadata = []
    test_metadata = []
    admin_metadata = []

    # If first batch, resets the metadata files and writes header
    csv_train_columns = ['filename', 'level', 'classtype', 'query', 'label']
    csv_test_columns = ['filename', 'level', 'classtype', 'query']
    if batch_idx == -1:
        with open(path+'TRAIN_metadata.csv', 'w') as outfile:
            dw = csv.DictWriter(outfile, delimiter=',', fieldnames=csv_train_columns) 
            dw.writeheader() 
        with open(path+'TEST_metadata.csv', 'w') as outfile:
            dw = csv.DictWriter(outfile, delimiter=',', fieldnames=csv_test_columns) 
            dw.writeheader() 
        with open(path+'ADMIN_metadata.csv', 'w') as outfile: 
            dw = csv.DictWriter(outfile, delimiter=',', fieldnames=csv_train_columns) 
            dw.writeheader() 
        return

    for data_class, mem in Dataclass.__members__.items():
        print("Starting generating data for: "+data_class)
        level_switcher = {
            # This dictionary matches each data class with a tuple, 
            # which includes (figure class, level in integer)
            'LENGTH': (LowLevelFigure, 1), 
            'LENGTHS': (LowLevelFigure, 1),
            'ANGLE': (LowLevelFigure, 1), 
            'ANGLES': (LowLevelFigure, 1),
            'SIMPLE_BAR': (MidLevelFigure, 2),
            'SIMPLE_PIE': (MidLevelFigure, 2),
            'ADVANCED_BAR': (HighLevelFigure,3), 
            'ADVANCED_PIE': (HighLevelFigure,3)
        }
        (figure_class, lev) = level_switcher.get(data_class)
        
        (TRAIN_DATA, TEST_DATA) = figure_class.generate_figures(data_class.lower(), counts=(data_counts['train'], data_counts['test']))
        # For high level data, we need to match queries to each figure
        if lev==3:
            (train_queries, train_labels) = advanced_query_matcher(TRAIN_DATA)
            (test_queries, test_labels) = advanced_query_matcher(TEST_DATA)

        print("Data generation completed. Now exporting to files")
        # Training dataset
        for i in range(len(TRAIN_DATA)):
            if (i%1000==999):
                print("Exporting " + str(i) + "th training image in data class "+data_class)
            # Exports image files
            td = TRAIN_DATA[i]
            im_filename = prefix+data_class+'_train_'+str(i)+'b'+str(batch_idx)+'.png'
            cv2.imwrite(path+im_filename, td[0])
            # Writing metadata
            if lev<3:
                # For lower level data, we directly add to the metadata
                mtdt = {"filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": SIMPLE_QUERIES[mem.value], "label":td[1].tolist()}
            elif lev==3:
                # For higher level data, we match it to the query
                mtdt = {"filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": train_queries[i], 
                    "label":train_labels[i]} 
            train_metadata = train_metadata + [mtdt]

        # Testing dataset
        for i in range(len(TEST_DATA)): 
            # Exports image files
            td = TEST_DATA[i]
            im_filename = prefix+data_class+'_test_'+str(i)+'b'+str(batch_idx)+'.png'
            cv2.imwrite(path+im_filename, td[0])
            # Writing metadata
            if lev<3:
                # For lower level data, we directly add to the metadata 
                mtdt = {"filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": SIMPLE_QUERIES[mem.value]}
                amtdt = { # Only the admin metadata has the true label
                    "filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": SIMPLE_QUERIES[mem.value], 
                    "label":td[1].tolist()} 
            elif lev==3:
                # For higher level data, we match it to the query
                mtdt = {"filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": test_queries[i]}
                amtdt = { # Only the admin metadata has the true label
                    "filename": im_filename, "classtype": data_class.lower(), "level": lev, "query": test_queries[i], 
                    "label":test_labels[i]} 
                # And then add to metadata

            test_metadata = test_metadata + [mtdt]
            admin_metadata = admin_metadata + [amtdt]

    with open(path+'TRAIN_metadata.csv', 'a') as outfile:
        dw = csv.DictWriter(outfile, fieldnames=csv_train_columns)
        for d in train_metadata:
            dw.writerow(d)
    with open(path+'TEST_metadata.csv', 'a') as outfile:
        dw = csv.DictWriter(outfile, fieldnames=csv_test_columns)
        for d in test_metadata:
            dw.writerow(d)
    with open(path+'ADMIN_metadata.csv', 'a') as outfile: 
        dw = csv.DictWriter(outfile, fieldnames=csv_train_columns)
        for d in admin_metadata:
            dw.writerow(d)


