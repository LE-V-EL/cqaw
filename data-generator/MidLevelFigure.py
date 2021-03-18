### Code started 2021 Feb
import math
from collections import Counter 
import numpy as np
import os
import sys
import cv2
import enum

class MidLevelFigure: 
    # Basic parameters for image
    FigSize = (400, 400)
    FigCenter = (200, 200)
    bottomY = 20

    # For bars
    val_min = 32
    val_max = 320
    bar_width = 16
    # For pies
    pie_rad_fixed = 100
    pie_min = 28 # min of radius, for variable tasks
    pie_max = 120 # max of radius, for variable task.. 


    @staticmethod
    def generate_figures(data_class, counts=(8,2)):
        switcher = {
            'simple_bar' : MidLevelFigure.generate_figure_bar, 
            'simple_pie' : MidLevelFigure.generate_figure_pie
        }
        figFunc = switcher.get(data_class)

        ### We don't need to check for distribution anymore! 
        #def check_distribution(nums):
        #    # we dont care until we reach larger amounts
        #    if len(nums) < 1000:
        #        return True
        #    c = Counter(nums)
        #    # not adding anything over 110% of the mean amount in each angle bucket
        #    threshold = np.mean(list(c.values())) * 1.1
        #    for (k, v) in c.items():
        #        if int(v) > threshold:
        #            return False
        #    return True

        total_count = counts[0]+counts[1]
        num_of_numbers = np.random.randint(3, 7, total_count) # How many bars/pies each figure will have
        N = np.sum(num_of_numbers)

        flat = np.repeat(np.arange(MidLevelFigure.val_min,  MidLevelFigure.val_max+1), 
            math.ceil(N/(MidLevelFigure.val_max+1-MidLevelFigure.val_min)))
        np.random.shuffle(flat)

        trainNums = []
        testNums = []

        curIdx = 0
        for i in range(counts[0]):
            # Trims the flat array into array of arrays, 
            # FOR TRAINING SET
            trainNums = trainNums + [flat[curIdx:curIdx+num_of_numbers[i]]]
            curIdx = curIdx+num_of_numbers[i]

        for j in range(counts[0], counts[0]+counts[1]):
            # Trims the flat array into array of arrays, 
            # FOR TEST SET
            testNums = testNums + [flat[curIdx:curIdx+num_of_numbers[j]]]
            curIdx = curIdx+num_of_numbers[j]

        return ([figFunc(nums=trainNums[i], testFlag=False) for i in range(counts[0])], 
            [figFunc(nums=testNums[i], testFlag=True) for i in range(counts[1])])

    @staticmethod
    def generate_figure_bar(nums=None, testFlag=False): 
        im = np.ones(MidLevelFigure.FigSize, dtype=np.float32) 
        # All black-and-white pie chart
        if nums is None:
            nums = np.random.randint(MidLevelFigure.val_min, MidLevelFigure.val_max+1, size=(6,))
        else: 
            nums = np.array(nums)
        num = len(nums)
        # If test data, adds variability
        if testFlag: 
            lineWidth = 1 + np.random.randint(0,5); # in Linewidth
            Y = MidLevelFigure.FigSize[1] - MidLevelFigure.bottomY - np.random.randint(-10, 11) # in bar placements
            barWidth = MidLevelFigure.bar_width + np.random.randint(-2, 3)
        else: 
            lineWidth = 1
            Y = MidLevelFigure.FigSize[1] - MidLevelFigure.bottomY
            barWidth = MidLevelFigure.bar_width
        # X positions
        Xs = [int(np.round(MidLevelFigure.FigSize[0]/(num+1)))*(i+1) for i in range(num)]

        # Draws rectangles
        for i in range(num):
            cv2.rectangle(im, (Xs[i]-barWidth,Y), (Xs[i]+barWidth,Y-nums[i]), 0, lineWidth)

        # Draws the x-axis at the bottom
        cv2.line(im, (0, Y), (MidLevelFigure.FigSize[0], Y), 0, 1)

        # Adds noise and normalizes
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        
        # TODO: comment this later
        #tempText = np.array2string(nums)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)
        
        # TODO: uncomment this part when you are going to save the figure instead of showing
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        # Returns the figure and true labels
        return (im, nums)

    @staticmethod
    def generate_figure_pie(nums=None, testFlag=False):
        im = np.ones(MidLevelFigure.FigSize, dtype=np.float32) 
        # All black-and-white pie chart
        if nums is None:
            nums = np.random.randint(MidLevelFigure.val_min, MidLevelFigure.val_max+1, size=(6,))
        nums = nums/np.sum(nums)
        nums = np.around(nums, 3)
        num = len(nums)
        # If test data, adds variability
        if testFlag: 
            lineWidth = 1 + np.random.randint(0,5); # in Linewidth
            X = MidLevelFigure.FigCenter[0] + np.random.randint(-40, 44)
            Y = MidLevelFigure.FigCenter[1] + np.random.randint(-40, 44) # in pie placement 
            r = np.random.randint(MidLevelFigure.pie_min, MidLevelFigure.pie_max+1)
        else:
            lineWidth = 1
            X = MidLevelFigure.FigCenter[0]
            Y = MidLevelFigure.FigCenter[1]
            r = MidLevelFigure.pie_rad_fixed
 
        # Draws circle
        cv2.ellipse(im, (X, Y), (r, r), 0, 0, 360, 0, lineWidth)

        # Partitions each 'pie'
        HAFP = int(np.round(nums[0]*360/2)) # Half the Angle of First Pie, used temporarily
        QAFP = int(np.round(nums[0]*360/4)) # Quarter the Angle of First Pie, used temporarily
        first_line_dir = np.random.randint(90-HAFP-QAFP, 90-QAFP) # TODO: do something about this
        theta = (np.pi / 180.0) * first_line_dir
        END = (X - r*np.cos(theta), Y - r*np.sin(theta))
        cv2.line(im, (X,Y), (int(np.round(END[0])), int(np.round(END[1]))), 0, lineWidth)
        for i in range(num-1):
            second_line_dir = first_line_dir+(360*nums[i]) # Direction of two lines in an angle
            # draws first line            
            theta = (np.pi / 180.0) * second_line_dir
            END = (X - r*np.cos(theta), Y - r*np.sin(theta))
            cv2.line(im, (X,Y), (int(np.round(END[0])), int(np.round(END[1]))), 0, lineWidth)
            # updates first line dir
            first_line_dir = second_line_dir

        # TODO: comment this later
        #tempText = np.array2string(nums)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)
        
        # Adds noise and normalizes
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        # TODO: uncomment this part when you are going to save the figure instead of showing
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        # Returns the figure and true labels
        return (im, nums)


#(im, angleSize) = MidLevelFigure.generate_figure_pie(testFlag=True)
#print(angleSize)
#cv2.imshow("Testing angle",im)
#cv2.waitKey(0)
#cv2.imwrite('testtest.png', im)
