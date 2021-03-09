from LowLevelFigure import LowLevelFigure
from MidLevelFigure import MidLevelFigure
from HighLevelFigure import HighLevelFigure
import numpy as np
import os,sys
import cv2
import csv

class SimpleDataset:

    arr = []
    train_dats = [] # Array of tuples, each tuple is (image, label)
    test_dats = [] # Array length follows self.counts

    QUERIES = {'length': 'What is the length of the line in the figure?', 
            'lengths': 'What are the lengths of the lines in the figure, from left to right?', 
            'angle': 'What is the degree of the angle in the figure?',
            'angles': 'What are the degrees of the angles in the figure, from left to right?', 
            'simple_bar': 'What are the values represented in the figure?', 
            'simple_pie': 'What are the values represented in the figure, clockwise from the top?'}

    LEVELS = {'length': 1, 
            'lengths': 1, 
            'angle': 1,
            'angles': 1, 
            'simple_bar': 2, 
            'simple_pie': 2}

    def __init__(self, 
        counts = {"train":8, "test":20}, 
        data_class='angle'):
        self.counts = counts
        self.data_class = data_class
        self.query = self.QUERIES[self.data_class]
        self.level = self.LEVELS[self.data_class]

    def generate_images(self):
        if self.level == 1:
            self.train_dats = LowLevelFigure.generate_figures(self.data_class, size=self.counts['train'], testFlag=False)
            self.test_dats = LowLevelFigure.generate_figures(self.data_class, size=self.counts['test'], testFlag=True)
        elif self.level == 2:
            self.train_dats = MidLevelFigure.generate_figures(self.data_class, size=self.counts['train'], testFlag=False)
            self.test_dats = MidLevelFigure.generate_figures(self.data_class, size=self.counts['test'], testFlag=True)

    def export_files(self, path="./fig/",prefix='Sample_'):
        with open(path+prefix+'train_metadata.csv', 'a', newline='') as csvfile:
            fieldnames = ['filename', 'level', 'data_class', 'query', 'true_label']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #csv_writer.writeheader()

            for i in range(len(self.train_dats)):
                td = self.train_dats[i]
                # Writing to the image file 
                im_filename = path+prefix+self.data_class+'_train_'+str(i)+'.png'
                cv2.imwrite(im_filename, td[0])
                #   Writing the metadata 
                csv_writer.writerow({'filename': im_filename, 'level': self.level, 'data_class': self.data_class, 'query': self.query, 'true_label': td[1]})
                #TODO: append instead of writing from scratch

        with open(path+prefix+'test_metadata.csv', 'a', newline='') as csvfile:
            fieldnames = ['filename', 'level', 'data_class', 'query', 'true_label']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #csv_writer.writeheader()
            for i in range(len(self.test_dats)):
                td = self.test_dats[i]
                im_filename = path+prefix+self.data_class+'_test_'+str(i)+'.png'
                cv2.imwrite(im_filename, td[0])
                csv_writer.writerow({'filename': im_filename, 'level': self.level, 'data_class': self.data_class, 'query': self.query, 'true_label': td[1]})


class AdvancedDataset: 
    arr = []
    train_dats = [] # Array of tuples, each tuple is (image, label)
    test_dats = [] # Array length follows self.counts
    train_queries = []
    train_labels = []
    test_queries = []
    test_labels = []

    BAR_QUERIES = {0: 'What is the length of {X}?', 
        1: 'Which variety has the maximum length?', 
        2: 'Which variety has the minimum length?', 
        3: 'Is {X} bigger than {Y}?', 
        4: 'Is {X} smaller than {Y}?'}

    PIE_QUERIES = {0: 'What is the value of {X}?', 
        1: 'Which variety has the maximum value?', 
        2: 'Which variety has the minimum value?', 
        3: 'Is {X} bigger than {Y}?',
        4: 'Is {X} smaller than {Y}?'}

    def __init__(self, 
        counts = {"train":5, "test":5}, 
        data_class='advanced_bar'):
        self.counts = counts
        self.data_class = data_class
        self.level = 3

    def generate_images(self):
        self.train_dats = HighLevelFigure.generate_figures(self.data_class, size=self.counts['train'], testFlag=False)
        self.test_dats = HighLevelFigure.generate_figures(self.data_class, size=self.counts['test'], testFlag=True)

    def query_matcher(self):
        # Matches images with queries and auto-label
        if self.data_class=='advanced_bar':
            self.query_matcher_bar()
        elif self.data_class=='advanced_pie':
            self.query_matcher_pie()
        else: 
            print('Query cannot be matched with current data class')

    def query_matcher_bar(self):
        # TODO: implement this! 
        print(self.BAR_QUERIES.keys())

    def query_matcher_pie(self):
        # TODO: implement this! 
        print(self.PIE_QUERIES.keys())

    def export_files(self, path="./fig/",prefix='Sample_'):
        with open(path+prefix+'train_metadata.csv', 'a', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for i in range(len(self.train_dats)):
                td = self.train_dats[i]
                # Writing to the image file 
                im_filename = path+prefix+self.data_class+'_train_'+str(i)+'.png'
                cv2.imwrite(im_filename, td[0])

        with open(path+prefix+'test_metadata.csv', 'a', newline='') as csvfile:
            #csv_writer.writeheader()
            for i in range(len(self.test_dats)):
                td = self.test_dats[i]
                im_filename = path+prefix+self.data_class+'_test_'+str(i)+'.png'
                cv2.imwrite(im_filename, td[0])


### Generation script sample
#d = SimpleDataset(data_class='simple_bar')
#d.generate_images()
#d.export_files(path='./test_fig/')

# TODO: Header should be added manually to each csv file
d = AdvancedDataset(data_class='advanced_pie')
d.generate_images()
d.query_matcher()
#d.export_files(path='./test_fig/')