### Code started 2021 Feb
import math
import numpy as np
import os
import sys
import cv2
import enum

class LowLevelFigure: 
    # Basic parameters for image
    FigSize = (400, 400)
    FigCenter = (200, 200)

    # For lines
    line_min = 24
    line_max = 240 # Sticking to the Figure12.py value https://github.com/Rhoana/perception/blob/master/EXP/ClevelandMcGill/figure12.py
    line_x_positions=np.array([50, 150, 250, 350])
    line_y_positions=np.array([200, 200, 200, 200]) 
    # For angles
    angle_radius = 35
    angle_min = 10
    angle_DOF = 90
    angle_x_positions=np.array([50,150,250,350])
    angle_y_positions=np.array([140,180,220,260])


    @staticmethod
    def generate_figures(data_class, counts=(8,2)):
        switcher = { # Sampling function and Figure generation function for each data class paired together
            'length': (LowLevelFigure.sample_length,
                LowLevelFigure.generate_figure_length), 
            'angle': (LowLevelFigure.sample_angle, 
                LowLevelFigure.generate_figure_angle),
            'lengths': (LowLevelFigure.sample_lengths, 
                LowLevelFigure.generate_figure_lengths),
            'angles': (LowLevelFigure.sample_angles, 
                LowLevelFigure.generate_figure_angles)
        }
        (sampleFunc, figFunc) = switcher.get(data_class)

        trainArgs={'testFlag':False}
        testArgs={'testFlag':True} # The obvious flags

        (trainNums, testNums) = sampleFunc(counts) # a tuple of train and test values, in a tuple of arrays
        trainArgs['nums'] = trainNums
        testArgs['nums'] = testNums

        # Converts the sampled values to cv2 images and returns in a tuple of arrays
        return ([figFunc(nums=trainNums[i], testFlag=False) for i in range(counts[0])], 
            [figFunc(nums=testNums[i], testFlag=True) for i in range(counts[1])])

    @staticmethod
    def sample_length(counts):
        # counts is a tuple: (size of training dataset, size of test dataset)
        total_count = counts[0]+counts[1]
        # Creates an array with a uniform # of integers [line_min, line_max]
        flat = np.repeat(np.arange(LowLevelFigure.line_min, LowLevelFigure.line_max+1), 
            math.ceil(total_count/(LowLevelFigure.line_max+1-LowLevelFigure.line_min)))
        # Shuffles the array
        np.random.shuffle(flat)
        trains = flat[:counts[0]]
        tests = flat[counts[0]:total_count]
        # Returns the array
        return (trains,tests)

    @staticmethod
    def sample_lengths(counts):
        # counts is a tuple: (size of training dataset, size of test dataset)
        total_count = counts[0]+counts[1]
        # Creates an array with a uniform # of integers [line_min, line_max]
        flat = np.repeat(np.arange(LowLevelFigure.line_min, LowLevelFigure.line_max+1), 
            math.ceil(total_count*4/(LowLevelFigure.line_max+1-LowLevelFigure.line_min)))
        #Shuffles the array and cuts into 4
        np.random.shuffle(flat)
        trains = flat[:counts[0]*4]
        tests = flat[counts[0]*4:total_count*4]
        return (trains.reshape((-1,4)),tests.reshape((-1,4)))

    @staticmethod
    def sample_angle(counts):
        # counts is a tuple: (size of training dataset, size of test dataset)
        total_count = counts[0]+counts[1]
        # Creates an array with a uniform # of integers [angle_min, angle_DOF]
        flat = np.repeat(np.arange(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF+1), 
            math.ceil(total_count/(LowLevelFigure.angle_DOF+1-LowLevelFigure.angle_min)))
        # Shuffles the array
        np.random.shuffle(flat)
        trains = flat[:counts[0]]
        tests = flat[counts[0]:total_count]
        # Returns the array
        return (trains,tests)
 
    @staticmethod
    def sample_angles(counts):
        # counts is a tuple: (size of training dataset, size of test dataset)
        total_count = counts[0]+counts[1]
        # Creates an array with a uniform # of integers [angle_min, angle_DOF]
        flat = np.repeat(np.arange(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF+1), 
            math.ceil(total_count*4/(LowLevelFigure.angle_DOF+1-LowLevelFigure.angle_min)))
        #Shuffles the array and cuts into 4
        np.random.shuffle(flat)
        trains = flat[:counts[0]*4]
        tests = flat[counts[0]*4:total_count*4]
        return (trains.reshape((-1,4)),tests.reshape((-1,4)))

    @staticmethod 
    def generate_figure_length(nums=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        lineLength = nums
        if lineLength is None:
            lineLength = np.random.randint(LowLevelFigure.line_min, LowLevelFigure.line_max)
        
        if lineLength%2 == 1:
            lineLength -= 1

        if testFlag: 
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            X = LowLevelFigure.FigCenter[0] + np.random.randint(-150, 150)
            Y = LowLevelFigure.FigCenter[1] + np.random.randint(-40, 40) # in Placement
            # Still sticking to the Figure12.py value
        else: 
            lineWidth = 1
            X = LowLevelFigure.FigCenter[0]
            Y = LowLevelFigure.FigCenter[1]

        # draws the line, starting from center to two directions
        cv2.line(im, (X, Y), (X, int(Y+lineLength/2)), 0, lineWidth)
        cv2.line(im, (X, Y), (X, int(Y-lineLength/2)), 0, lineWidth)

        # TODO: comment this later
        #tempText = str(lineLength)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)

        # adds noise
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        return (im, lineLength)

    # Returns a figure with multiple lines and their lengths in numpy array
    @staticmethod
    def generate_figure_lengths(nums=None, y_positions=None, testFlag=False): # TODO: do something about other "flags" later
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        lengths = nums
        if lengths is None:
            lengths = np.random.randint(LowLevelFigure.line_min, LowLevelFigure.line_max, size=(4,))
            lengths = np.rint(lengths/2)*2
        if y_positions is None: 
            y_positions = LowLevelFigure.line_y_positions

        if testFlag:
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            x_positions = LowLevelFigure.line_x_positions + np.random.randint(-5, 5, size=(4,))# in x-axis placement ("wiggle")
            y_positions = y_positions + np.random.randint(-10, 10, size=(4,))
        else:
            lineWidth = 1
            x_positions = LowLevelFigure.line_x_positions

        for i in range(len(lengths)):
            # draws each line
            cv2.line(im, (x_positions[i], y_positions[i]), (x_positions[i], y_positions[i]-int(lengths[i]/2)), 0, lineWidth)
            cv2.line(im, (x_positions[i], y_positions[i]), (x_positions[i], y_positions[i]+int(lengths[i]/2)), 0, lineWidth)

        # TODO: comment this later
        #tempText = np.array2string(lengths)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)
        
        # adds noise
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        return (im, lengths)


    # Returns a figure with a single angle and the angle size, in tuple
    @staticmethod
    def generate_figure_angle(nums=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        angleSize = nums
        if angleSize is None:
            angleSize = np.random.randint(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF)
        if testFlag:
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,4); # in Linewidth
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

        # TODO: comment this later
        #tempText = str(angleSize)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)
        
        # adds noise
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        return (im, angleSize)


    @staticmethod
    def generate_figure_angles(nums=None, y_positions=None, testFlag=False):
        im = np.ones(LowLevelFigure.FigSize, dtype=np.float32) 
        angles = nums
        if angles is None:
            angles = np.random.randint(LowLevelFigure.angle_min, LowLevelFigure.angle_DOF, size=(4,))
        if y_positions is None: 
            y_positions = LowLevelFigure.angle_y_positions

        if testFlag: 
            # If test data, then add variability
            lineWidth = 1 + np.random.randint(0,3); # in Linewidth
            x_positions = LowLevelFigure.line_x_positions + np.random.randint(-5, 5, size=(4,))# in x-axis placement ("wiggle")
            y_positions = y_positions + np.random.randint(-10, 10, size=(4,))
        else:
            lineWidth = 1
            x_positions = LowLevelFigure.angle_x_positions

        for i in range(len(angles)):
            angleSize = angles[i]
            X = x_positions[i]
            Y = y_positions[i]

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

        # TODO: comment this later
        #tempText = np.array2string(angles)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(im, tempText, (150, 15), font, 0.4, 0.3)

        # adds noise
        noise = np.random.uniform(0, 0.05, im.shape)
        im += noise
        im = im*255.0
        im = np.minimum(im, 255.0)
        im = np.maximum(im, 0.0)

        return (im, angles)



### Quick script for testing LowLevelFigure class
#(im, angleSize) = LowLevelFigure.generate_figure_angles(testFlag=True)
#print(angleSize)
#cv2.imshow("Testing length",im)
#cv2.waitKey(0)

