import math
import numpy as np
import os
import sys
import cv2
import enum

class Data_Classes(enum.Enum): 
    length = 1
    angle = 2
    lengths = 3
    angles = 4

class LowLevelFigure: 

    # Basic parameters for image
    FigSize = (100, 100)
    FigCenter = (50, 50)
    # For lines
    line_min = 6
    line_max = 60 # Sticking to the Figure12.py value https://github.com/Rhoana/perception/blob/master/EXP/ClevelandMcGill/figure12.py
    line_x_positions=np.array([15, 35, 55, 75])
    line_y_positions=np.array([50, 50, 50, 50]) # TODO: What should we do with this? 
    # For angles
    angle_radius = 10
    angle_min = 10
    angle_DOF = 90

    @staticmethod
    def generate_data():
        ### There are only so many different lengths (12, since all are even) and angles (80) we can create, 
        ### so for single-element figures I think it makes sense to simply shuffle the order of a preexisting values 
        ### and not check for the duplicates. 

        ### For four lengths/angles in one figure, 
        ### can the duplicate test check for the same length/angle in the same index in another figure? 

        for c in Data_Classes: 
            pass # TODO: edit this
            
        return arr_shuffled

    @staticmethod 
    def generate_figure_length(lineLength=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        if lineLength is None:
            lineLength = np.random.randint(LowLevelFigure.line_min, LowLevelFigure.line_max)
        
        if lineLength%2 == 1:
            lineLength -= 1

        if testFlag: 
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            X = LowLevelFigure.FigCenter[0] + np.random.randint(-45, 45)
            Y = LowLevelFigure.FigCenter[1] + np.random.randint(-10, 10) # in Placement
            # Still sticking to the Figure12.py value
        else: 
            lineWidth = 1
            X = LowLevelFigure.FigCenter[0]
            Y = LowLevelFigure.FigCenter[1]

        # draws the line, starting from center to two directions
        cv2.line(im, (X, Y), (X, int(Y+lineLength/2)), 0, lineWidth)
        cv2.line(im, (X, Y), (X, int(Y-lineLength/2)), 0, lineWidth)

        # adds noise
        noise = np.random.uniform(0, 0.05, (100, 100))
        im += noise

        return (im, lineLength)

    # Returns a figure with multiple lines and their lengths in numpy array
    @staticmethod
    def generate_figure_lengths(lengths=None, y_positions=None, testFlag=False): # TODO: do something about other "flags" later
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        if lengths is None:
            lengths = np.random.randint(LowLevelFigure.line_min, LowLevelFigure.line_max, size=(4,))
            lengths = np.rint(lengths/2)*2
        if y_positions is None: 
            y_positions = LowLevelFigure.line_y_positions+np.random.randint(-5, 5, size=(4,))

        if testFlag:
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            x_positions = LowLevelFigure.line_x_positions + np.random.randint(-3, 3, size=(4,))# in x-axis placement ("wiggle")
        else:
            lineWidth = 1
            x_positions = LowLevelFigure.line_x_positions

        for i in range(len(lengths)):
            # draws each line
            cv2.line(im, (x_positions[i], y_positions[i]), (x_positions[i], y_positions[i]-int(lengths[i]/2)), 0, lineWidth)
            cv2.line(im, (x_positions[i], y_positions[i]), (x_positions[i], y_positions[i]+int(lengths[i]/2)), 0, lineWidth)

        # adds noise
        noise = np.random.uniform(0, 0.05, (100, 100))
        im += noise

        return (im, lengths)


    # Returns a figure with a single angle and the angle size, in tuple
    @staticmethod
    def generate_figure_angle(angleSize=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        if angleSize is None:
            angleSize = np.random.randint(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF)

        if testFlag:
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            X = LowLevelFigure.FigCenter[0] + np.random.randint(-30, 30)
            Y = LowLevelFigure.FigCenter[1] + np.random.randint(-30, 30) # in Placement 
        else:
            lineWidth = 1
            X = LowLevelFigure.FigCenter[0]
            Y = LowLevelFigure.FigCenter[1]

        first_line_dir = np.random.randint(360)
        second_line_dir = first_line_dir+angleSize # Direction of two lines in an angle

        # draw first line            
        theta = -(np.pi / 180.0) * first_line_dir
        END = (X - LowLevelFigure.angle_radius * np.cos(theta), Y - LowLevelFigure.angle_radius * np.sin(theta))
        cv2.line(im, (X,Y), (int(np.round(END[0])), int(np.round(END[1]))), 0, lineWidth)

        # draw second line
        theta =  -(np.pi / 180.0) * second_line_dir
        END = (X - LowLevelFigure.angle_radius * np.cos(theta), Y - LowLevelFigure.angle_radius * np.sin(theta))
        cv2.line(im, (X,Y), (int(np.round(END[0])), int(np.round(END[1]))), 0, lineWidth)

        # adds noise
        noise = np.random.uniform(0, 0.05, (100, 100))
        im += noise

        return (im, angleSize)

(im, angleSize) = LowLevelFigure.generate_figure_lengths(testFlag=True)
print(angleSize)
cv2.imshow("Testing angle",im)
cv2.waitKey(0)