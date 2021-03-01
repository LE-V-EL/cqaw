import math
import numpy as np
import os
import sys
import cv2

class LowLevelFigure: 

    # Basic parameters for image
    FigSize = (100, 100)
    FigCenter = (50, 50)
    # For lines
    line_min = 6
    line_max = 40
    # For angles
    angle_radius = 10
    angle_min = 10
    angle_DOF = 90

    @staticmethod 
    def generate_figure_line(lineLength=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        if lineLength is None:
            lineLength = np.random.randint(LowLevelFigure.line_min, LowLevelFigure.line_max)
        
        if lineLength%2 == 1:
            lineLength -= 1

        if testFlag: 
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            X = LowLevelFigure.FigCenter[0] + np.random.randint(-45, 45)
            Y = LowLevelFigure.FigCenter[1] + np.random.randint(-20, 20) # in Placement
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


    # Returns a figure with a single angle and the angle size, in tuple
    @staticmethod
    def generate_figure_angle(angleSize=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        if angleSize is None:
            angleSize = np.random.randint(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF+1)

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

(im, angleSize) = LowLevelFigure.generate_figure_angle(40, True)
print(angleSize)
cv2.imshow("Testing angle",im)
cv2.waitKey(0)