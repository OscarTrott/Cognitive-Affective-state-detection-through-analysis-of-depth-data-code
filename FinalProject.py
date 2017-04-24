import numpy as np
import time
import sys
import os
import math
import shutil
from matplotlib import pyplot as pp
import tkinter
import cv2
import h5py
#Commented out as unnecessary when processing 
#import pykinect2
#from pykinect2 import PyKinectRuntime
import threading
import glob
import random
from collections import deque

from datetime import datetime

import _thread as thread

class View(object):
    """Simple video control gui"""
    playing = True

    def fastForward(self):
        """Fast forwards video playback"""
        self.model.fastForward()

    def rewind(self):
        """Rewinds video playback"""
        self.model.rewind()

    def playPause(self):
        """Function linked to play/pause button, plays and pauses execution"""
        if self.playing:
            self.model.fpause()
            self.p.config(text="play")
        else:
            self.model.fplay()
            self.p.config(text="Pause")

        self.playing = not self.playing
            
    def stop(self):
        """Function linked to stop button, stops the current execution"""
        self.model.file.running = False
    
    def updateTime(self, time):
        """Updates the time shown on the GUI"""
        self.timeLbl.config(text='Time: '+time[0]+' : '+time[1]+' : '+time[2])

    def initWin(self, model):
        """
        Initialises the GUI

        :param model: the processor model to which the gui is linked
        """
        self.root = tkinter.Tk()
        self.model = model
        r=tkinter.Button(self.root, text="Rewind", command=self.rewind)
        self.p=tkinter.Button(self.root, text = "Pause", command=self.playPause)
        f=tkinter.Button(self.root, text="Fast forward", command=self.fastForward)
        s=tkinter.Button(self.root, text="Stop", command=self.stop)
        self.timeLbl = tkinter.Label(self.root, text="Time:0:00:00")

        r.pack()
        self.p.pack()
        f.pack()
        s.pack()
        self.timeLbl.pack()

        self.root.update()
        self.root.deiconify()

        self.root.mainloop()


class Hdf5FileHandler(object):
    """HDF5 file handler class, allows simple writing to and from our storage files"""
    fileName = ""
    segmentNum = -1
    f = None
    data = []   
    segments = []
    
    def __init__(self, model):
        """Returns an HDF5 file handler linked to the given processor"""
        self.model = model

    def setAttribute(self, dataset, value, name):
        """
        Sets the dataset "dataset" to have an attribute named "name" to have value "value" 
        :param dataset: if None then the current dataset is set to have the attribute
        :param value: if "file" then the last used file shall be set to have the attrtibute value
        :param name: identifier of attribute being modified
        """

        if dataset == "file":
            self.f.attrs[name] = value
        elif dataset is None:
            self.f["footage/"+self.currentType].attrs[name] = value
        elif dataset in f:
            self.f[dataset].attrs[name] = value
        else:
            return None

    def getAttribute(self, dataset, name):
        """
        Returns the value in the specified attribute
        :param dataset: if None then the current dataset is set to have the attribute
        :param name: identifier of attribute being modified
        """
        if dataset == "file":
            if name in self.f.attrs:
                return self.f.attrs[name]
            else:
                return False
        elif dataset is None:
            if name in self.f["footage/"+self.currentType].attrs:
                return self.f["footage/"+self.currentType].attrs[name]
            else:
                return False
        elif dataset in self.f:
            if name in self.f[dataset].attrs:
                return self.f[dataset].attrs[name]
            else:
                return False
        else:
            return False
        
    def resetCount(self, footageType):
        """
        Resets the current frame/analysis count for a given dataset
        :param footageType: id of dataset having count reset
        """
        if footageType is None:
            self.options[self.currentType] = -1
            return
        elif not footageType in self.options:
            print("ERROR, unknown type: "+ footageType)
            return None

        self.options[footageType] = 0

    def setFile(self, read, **kwargs):
        """
        Sets the file being used and initialises base values
        :param read: sets whether the file is being written to or read from
        """
        self.max = 0
        self.segmentNum = -1
        self.firstSegment = True
        skipUser = False
        #Declares valid recording types and records into the corresponding dataset
        self.options = {"raw":-1,"noise_fix":-1,"noise_fix_2":-1,"noise_fix_3":-1,"bgdest":-1,"complete":-1,"movementEst":-1,"frameDif":-1,"opticalFlow":-1}
        #Sets the recording frame rate, computing and IO speed should be taken into accoutn when setting this
        self.fps = 20
        self.timeArray = np.ndarray((1,512))
        self.timeArray = self.timeArray.astype('uint16')
        self.timeArray[0,4] = self.fps
        self.running = True
        

        for key in kwargs:
            if key == "file":
                dirname=kwargs["file"]
                skipUser = True
                self.currentType = "raw"
        
        if not skipUser:

            
            print("Please enter file name")
            dirname = str(input())+".hdf5"
            
            self.f = h5py.File(dirname, 'a')
            print("Opening file")
            print("\nPlease enter the data type to be " + "read" if read else "written")

            #print(*options, sep='\n-')
            print ("\n-".join(self.options))
         
            inp = input()
        
            #Ends if user enters invalid choice
            if not inp=="" and not inp in self.options:
                print("Invalid choice")
                return False
            elif not inp=="":
                self.currentType = inp
        else:
            inp = "noise_fix_2"
            self.f = h5py.File(dirname, 'a') 
        
        #if "analysis/opticalFlow" in self.f:
        #    del self.f["analysis/opticalFlow"]
        if not inp == "":
            if not read:
                #Counts how many frames have been recorded in the past second
                self.numberOfFramesInSecond = 0

                #Checks if there is some footage in the file which will be overwritten
                if "footage/"+inp in self.f:
                    print("footage/"+inp)
                    self.data = self.f.get("footage/"+inp)
                    print("Some footage already exists, are you sure that you want to continue?")
                    print("y/n")
                    i = input()
                    if i == 'n':
                        return False
                    print("The data will be overwritten")
                    del self.f["footage/"+inp]
                    self.data = self.f.create_dataset("footage/"+inp, (0,425,512), dtype="int16", maxshape=(None, 425, 512), chunks=True)
                else:
                    self.data = self.f.create_dataset("footage/"+inp, (0,425,512), dtype="int16", maxshape=(None, 425, 512), chunks=True)
            else:
                #Checks if there is some footage in the file which 
                if "footage/"+inp in self.f:
                    self.data = self.f.get("footage/"+inp)
                else:
                    print("footage/"+inp+" does not exist in this file")
                    return False
                fps = 1/self.data[0,0,4]
                print("Reading "+inp+" from "+self.f.filename) 
            return True

    def getAnalysis(self, type):
        if "analysis/"+type in self.f:
            return self.f["analysis\\"+type]
        else:
            return None

    def getLabels(self, type):
        if "label/"+type in self.f:
            return self.f["label\\"+type]
        else:
            return None

    def setNextSegment(self, segment):
        """
        Adds the passed segment data to the file at the next point
        :param segment: should be a single dimension array of size 4 containing:
                1.Mean
                2.Range
                3.Standard deviation
                4.Entropy
        """
        self.segmentNum += 1

        if self.firstSegment:
            self.segments = self.f.create_dataset("analysis/segments", (0,4), maxshape=(None,4), dtype="float32", chunks=True, compression="lzf")

        self.segments.resize((self.segments.shape[0]+1,4))

        self.segments[self.segmentNum] = segment

    def getNextSegment(self):
        """
        Returns the next segment from the segment file
        """
        self.segmentNum += 1
        
        if self.firstSegment:
            self.segments = self.getAnalysis("segments")
            self.firstSegment = False

        #Returns frame if there is one, null if out of footage bounds            
        if self.segmentNum < self.data.shape[0] and self.frameNum >= 0:
            return self.data[self.frameNum]
        else:
            return None

    def getName(self):
        """Returns the filename including extension"""
        return self.f.filename

    def getRange(self):
        """Returns an array of values at loc 0,5 indicating movement detections"""
        buffer = 1
        a = self.f["footage/bgdest"]
        min_ = self.options["bgdest"]
        min_ = max(min_-buffer,0)
        max_ = self.options["bgdest"]
        max_ = min(max_+buffer,a.shape[0])
        a = a[min_:max_]
        
        a = a[:,0,5]
        return a

    def getFrameLength(self, footageType):
        """
        Returns the number of frames in the footage
        :param footageType: footage id of which the length is returned
        """
        if footageType == None:
            footageType = self.currentType
        elif not footageType in self.options:
            return None

        return self.f["footage/"+footageType].shape[0]

    def getFootageMax(self, footageType):
        """
        Returns the maximum value encountered within the footage
        :param footageType: id of dataset being analysed
        """
        if footageType == None:
            footageType = self.currentType
        elif not footageType in self.options:
            return None
        if self.max == 0:
            self.max = np.max(self.f["footage/"+footageType])
        return self.max

    def trim(self, footageType, start, end):
        """
        Trims the given dataset to the passed in values
        :param footageType: the dataset being trimmed
        :param start: the start to trim to
        :param end: the end to trim to
        """
        if footageType == None:
            footageType = self.currentType
        elif not footageType in self.options:
            return None

        if start > 0 and start <= end and end <= self.f["footage/"+footageType].shape[0]:
            temp = self.f["footage/"+footageType][start:end].copy()
            self.deleteFootage(footageType)
            self.f.create_dataset("footage/"+footageType, (temp.shape[0],425,512), dtype="int16", maxshape=(None, 425, 512), chunks=True)
            print(temp.shape)
            for i in range(temp.shape[0]):
                self.f["footage/"+footageType][i] = temp[i]
        else:
            return None

    def frameToFile(self, footageType, frame, **kwargs):
        """
        Appends the frame to the given dataset
        :param footageType: dataset to which the frame is added
        :param frame: frame to be added
        """
        #Updates data and footageType counter
        if footageType in self.options:
            self.options[footageType] += 1
        else:
            return None

        if not "footage/"+footageType in self.f or self.options[footageType]==0 :
            if "footage/"+footageType in self.f:
                del self.f["footage/"+footageType]
            self.data = self.f.create_dataset("footage/"+footageType, (0,frame.shape[0],frame.shape[1]), dtype="int16", maxshape=(None, frame.shape[0], frame.shape[1]), chunks=True)
            #self.data = self.f["footage/"+footageType]
        #else:
        #    self.data = self.f.create_dataset("footage/"+footageType, (0,425,512), dtype="int16", maxshape=(None, 425, 512), chunks=True, compression="lzf")
        straightCopy = False

        for key in kwargs:
            if key == "straightCopy" and kwargs[key]==True:
                straightCopy = True

        #A straight copy frame will have the time data alraeady included within its structure        
        if not straightCopy:
            #Set the time in milliseconds and give the frame rate
            self.timeArray[0,0] = datetime.now().hour
            self.timeArray[0,1] = datetime.now().minute
            self.timeArray[0,2] = datetime.now().second
            self.timeArray[0,3] = datetime.now().microsecond//1000

            frame = np.append(self.timeArray,frame,axis=0)

        self.f["footage/"+footageType].resize((self.data.shape[0]+1,self.data.shape[1],self.data.shape[2]))

        self.f["footage/"+footageType][self.options[footageType]] = frame

    def frameFromFile(self, footageType, frameSkip, **kwargs):
        """
        Adds the passed frame to the specified footage type
        
        :param footageType: dataset from which the frame is sourced
        :param frameSkip: number of frames to be skipped when sourcing the frame

        kwargs:
        "frameLocation" = Integer, gets the frame from the specified location within the file, returns None if it is out of bounds
        """
        if footageType in self.options:
            self.options[footageType] += frameSkip
        elif footageType == None:
            footageType = self.currentType
            self.options[footageType] += frameSkip
        else:
            return None

        if "frameLocation" in kwargs:
            oldPt = self.options[footageType]
            self.options[footageType] = kwargs["frameLocation"]

        if not self.currentType == footageType:
            self.data = self.f["footage/"+footageType]
            self.currentType = footageType
        #Returns frame if there is one, null if out of footage bounds            
        if self.options[footageType] < self.f["footage/"+footageType].shape[0] and self.options[footageType] >= 0 and self.running:
            #self.model.display(self.data[self.options[footageType]],1)
            result = self.f["footage/"+footageType][self.options[footageType]]
            if "frameLocation" in kwargs:
                self.options[footageType] = oldPt
            return result
        else:
            if "frameLocation" in kwargs:
                self.options[footageType] = oldPt
            return None

    def analysisToFile(self, movementEst, analysisType):
        """
        Stores the result of any given analysis performed
        :param analysisType: type of analysis which is being stored
        :param movementEst: should be a one dimensional array of length 4, where element:
                1. is the average frequency density of the histogram of the current frame, this should ideally be depth independent
                2. is the standard deviation of values which have moved
                3. is the duration, in frames, of the movement
                4. is the range of the number of pixels in each frame, e.g the difference between the number of pixels in the frame with the highest movement and the frame with the lowest movement
        """
        
        if not analysisType in self.options:
            return

        #First call in this analysis, overwrites by default
        if self.options[analysisType] == -1:
            if "analysis/"+analysisType in self.f:
                if analysisType == "opticalFlow":
                    self.f["analysis/"+analysisType].resize((0,movementEst.shape[0]))
                else:
                    self.f["analysis/"+analysisType].resize((0,movementEst.shape[0]))
            else:
                if analysisType == "opticalFlow":
                    self.f.create_dataset("analysis/"+analysisType,(0,movementEst.shape[0]),"float32",maxshape=(None,None))
                else:
                    self.f.create_dataset("analysis/"+analysisType,(0,movementEst.shape[0]),"float32",maxshape=(None,None))
        self.options[analysisType] += 1
        
        self.f["analysis/"+analysisType].resize((self.f["analysis/"+analysisType].shape[0]+1,self.f["analysis/"+analysisType].shape[1]))
        self.f["analysis/"+analysisType][self.options[analysisType]] = movementEst

    def analysisFromFile(self, analysisType):
        """
        Retrieves the result of any given analysis performed
        :param analysisType: type of analysis which is being retrieved
        :returns: a one dimensional array of length 4, where element:
                1. is the average frequency density of the histogram of the current frame, this should ideally be depth independent
                2. is the standard deviation of values which have moved
                3. is the duration, in frames, of the movement
                4. is the range of the number of pixels in each frame, e.g the difference between the number of pixels in the frame with the highest movement and the frame with the lowest movement
        """

        if not analysisType in self.options:
            return
        if not "analysis/"+analysisType in self.f:
            return np.array([])

        analysis = self.f["analysis/"+analysisType]
        out = np.zeros((analysis.shape[0],analysis.shape[1]+1))
        if analysisType=="opticalFlow": 
            out[...,0:analysis.shape[1]] = analysis
        else:
            out[...,0:analysis.shape[1]] = analysis
        #print(np.array(analysis))
        for i in range(analysis.shape[0]):
            if analysisType == "opticalFlow":
                out[i,5] = analysis[i,0]/analysis[i,4]
            else:
                out[i,4] = analysis[i,0]/analysis[i,2]
        return out[...]
                
    def deleteFootage(self, footageType):
        """
        Deletes the given footage dataset from the fiel, it should be noted that HDF5 does not support true deletion and as such the memory is not released
        """
        if footageType in self.options:
            self.options[footageType] = -1
        elif footageType == None:
            footageType = self.currentType
            self.options[footageType] += frameSkip
        else:
            return None

        del(self.f["footage/"+footageType])
 
    def deleteAnalysis(self, analysisType):
        """
        Deletes the given analysis dataset from the fiel, it should be noted that HDF5 does not support true deletion and as such the memory is not released
        """
        if analysisType in self.options:
            self.options[analysisType] = -1
        else:
            return None
        if "analysis/"+analysisType in self.f:
            del(self.f["analysis/"+analysisType])

    def convert(self, file2):
        """
        Transfers select datasets and analysis from this file into the passed file, used to release memory
        :param file2: the file to which all data is transferred
        """
        if "footage/noise_fix_2" in file2.f:
            del file2.f["footage/noise_fix_2"]
        if "footage/bgdest" in file2.f:
            del file2.f["footage/bgdest"]
        if "analysis" in file2.f:
            del file2.f["analysis"]
            file2.f.create_group("analysis")

        file2.f.create_dataset("footage/noise_fix_2",self.f["footage/noise_fix_2"].shape,self.f["footage/noise_fix_2"].dtype,maxshape=(None, self.f["footage/noise_fix_2"].shape[1], self.f["footage/noise_fix_2"].shape[2]), chunks=True)
        t = self.f["footage/noise_fix_2"][...]
        file2.f["footage/noise_fix_2"][...] = t
        file2.f.create_dataset("footage/bgdest",self.f["footage/bgdest"].shape,self.f["footage/bgdest"].dtype,maxshape=(None, self.f["footage/bgdest"].shape[1], self.f["footage/bgdest"].shape[2]), chunks=True)
        t = self.f["footage/bgdest"][...]
        file2.f["footage/bgdest"][...] = t

        if "analysis/movementEst" in self.f:
            file2.f.create_dataset("analysis/movementEst",self.f["analysis/movementEst"].shape,self.f["analysis/movementEst"].dtype,maxshape=(None, self.f["analysis/movementEst"].shape[1]), chunks=True)
            t = self.f["analysis/movementEst"][...]
            file2.f["analysis/movementEst"][...] = t
        file2.f.create_dataset("analysis/opticalFlow",self.f["analysis/opticalFlow"].shape,self.f["analysis/opticalFlow"].dtype,maxshape=(None, self.f["analysis/opticalFlow"].shape[1]), chunks=True)
        t = self.f["analysis/opticalFlow"][...]
        file2.f["analysis/opticalFlow"][...] = t

        file2.close()

    def transfer(self, file2):
        """
        Transfers all datasets and analysis from this file into the passed file, used to release memory
        :param file2: the file to which all data is transferred
        """
        for dataset in self.f:
            print(dataset)
            if dataset in file2.f:
                del file2.f[dataset]
            a = file2.f.create_dataset(dataset,self.f[dataset].shape,maxshape=(None,None),chunks=True)
            a.attrs["class"] = self.f[dataset].attrs["class"]
            b = self.f[dataset][...]
            print(b)
            a[...] = b


    def getFile(self):
        """Returns the file being accessed"""
        return self.f

    def getDatTime(self):
        """Returns current date time information from file"""
        return datetime(0,0,0,self.data[0,0,0],self.data[0,0,1],self.data[0,0,2],self.data[0,0,3])

    def close(self):
        """Closes file and releases resources, throws error if called twice without setFile being called"""
        self.f.close() 


class Cluster(object):
    """Class representing a flow field cluster"""
    def __init__(self, centre_, size, mag, num, predClust, **kwargs):
        """
        Returns a new cluster object with given parameters

        :param centre_: coordinates of cluster center
        :param size: current size in number of contained flow vectors
        :param mag: current average magnitude of contained flow vectors
        :param num: cluster number
        :param predClust: next predicted location
        """
        self.centre = np.array([centre_])
        self.size = np.array([size])
        self.magArray = np.array([mag])
        self.clustNumArr = np.array([num])
        self.predClustArr = np.array([predClust])
        self.duration = 0
        self.new = True
        self.speed = 0
        self.acceleration = 0
        self.accelerationSum = np.array([])
        self.regularisationX = 0
        self.regularisationY = 0

    def update(self, center, size, mag, num, predClust):
        """
        Updates the cluster, indicates that the movement is still occurring
        :param center: current center of cluster
        :param size: current size in number of contained flow vectors
        :param mag: current average magnitude of contained flow vectors
        :param num: cluster number
        :param predClust: next predicted location
        """
        centre_ = np.array([center])
        speed_ = np.linalg.norm(centre_ - self.centre)
        if self.speed != 0:
            self.acceleration = abs(self.speed - speed_)
            self.accelerationSum = np.append(self.accelerationSum, self.acceleration)
        self.speed = speed_
        self.duration += 1
        self.size = np.append(self.size, size)
        self.centre = np.append(self.centre, centre_,0)
        self.magArray = np.append(self.magArray, [mag], 0)
        self.clustNumArr = np.append(self.clustNumArr,[num])
        self.predClustArr = np.append(self.predClustArr,[predClust],0)
        self.renew()

    def getPos(self):
        """Returns current coordinates of cluster"""
        return self.centre[-1]

    def getNextPos(self):
        """Returns coordinates of next estimated position"""
        if self.duration<2:
            p = self.centre[-1]            
            return (p[0],p[1])
        
        p2 = self.centre[-1]
        p1 = self.centre[-2]

        if self.duration>2:
            p0 = self.centre[-3]
            self.regularisationX = 0.5*(p1[0] - p0[0]) + 0.5*self.regularisationX
            self.regularisationY = 0.5*(p1[1] - p0[1]) + 0.5*self.regularisationY

        dX = p2[0] - p1[0] + self.regularisationX
        dY = p2[1] - p1[1] + self.regularisationY

        return (p2[0]+dX,p2[1]+dY)
    
    def isNew(self):
        """Returns whether the cluster has been updated recently"""
        return self.new

    def renew(self):
        """Sets that the cluster has been updated recently"""
        self.new = True

    def getAge(self):
        """Returns number of frames for which the cluster has existed"""
        return self.duration

    def age(self):
        """Sets the cluster to be none new, it has not been updated recently"""
        self.new = False


class Classifier(object):
    """Class handles the classification task"""
    def __init__(self):
        """Creates the classifier object with default parameters and starts task"""

        #Opens relevant files
        classifications = h5py.File("classifications.hdf5")
        classCount = np.array([])
        valueArr = np.array([])
        self.classifications = classifications
        #Corrupt datasets
        self.invalids = ["inf8y115","ipsk11y115","zum_y115","nq1t10y113","zut9y113","zuty117"]
        instanceAcc = {key:(0,0) for key in classifications}
        
        #Parameters for classifier and results
        k = 3
        runs = 40
        correct = 0
        totInstanceNum = 0
        classCorrect = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
        
        #Generate accuracy modifiers for dataset
        accuracyWeightings = 6*np.unique(classCount,return_counts=True)[1]/(classCount.shape[0])
        self.probDensityDist = np.unique(classCount,return_counts=True)[1]/classCount.shape[0]

        #Split according to k
        splits = np.linspace(0,len(classifications),k+1)
        experimentNum = len(classifications)
        experiments = np.array([id for id in classifications.keys()])
        totAccuracy = 0
        
        #Classify dataset j*k times
        for j in range(runs):
            #Random shuffle of dataset
            np.random.shuffle(experiments)
            for i in range(splits.shape[0]-1):

                #Identify instances within the splits
                accuracy = 0
                if round(splits[i]) == 0:
                    training = experiments[int(round(splits[i+1])):]
                elif round(splits[i+1]) == round(splits[splits.shape[0]-1]):
                    training = experiments[:int(round(splits[i]))]
                else:
                    training = np.append(experiments[:int(round(splits[i]))],experiments[int(round(splits[i+1])):])
                testing = experiments[int(round(splits[i])):int(round(splits[i+1]))]

                self.training = training
            
                instanceNum = 0
                #Classify instances
                for test in testing:
                    if not test in self.invalids:
                        totInstanceNum += 1
                        instanceNum+=1

                        #Make prediction and get true class
                        data = classifications[test][...]
                        predClass = self.classify(data)
                        trueClass = classifications[test].attrs["class"]
                        dAcc = 0

                        #Update accuracy values using modified version
                        dAcc = 1/accuracyWeightings[predClass-1] if predClass == trueClass else 0

                        #Update accuracy values using true version
                        dAcc = 1 if predClass == trueClass else 0

                        #Update results
                        classCorrect[trueClass-1,1] = classCorrect[trueClass-1,1]+1
                        if predClass == trueClass:
                            correct += 1
                            classCorrect[predClass-1,0] = classCorrect[predClass-1,0]+1
                            instanceAcc[test] = (instanceAcc[test][0]+1,instanceAcc[test][1]+1)
                        else:
                            instanceAcc[test] = (instanceAcc[test][0],instanceAcc[test][1]+1)
                        
                        accuracy += dAcc
                    
                totAccuracy += (accuracy/instanceNum)/k
                print(totAccuracy/(j+1))
        totAccuracy = totAccuracy/runs
        #Output results to user
        print(classCorrect)
        print("Accuracy: "+str(totAccuracy))
        print("Total correct: "+str(correct)+" out of "+str(totInstanceNum))

    def classify(self, data):
        """Perform k-NN classification upon the data"""

        #k param
        k = 5
        dtype = [('dist',float),('id','U32')]
        q = np.empty((0,2),dtype=dtype)
        added = np.empty((1),dtype=dtype)
        classes = np.zeros([6])

        #Identify k nearest neighbours
        for instance in self.training:
            if not instance in self.invalids:
                #Get distance measure
                dist = self.distance(data, self.classifications[instance])            
            
                #update k-NN list
                if q.shape[0]<k:    
                    added['dist'] = dist
                    added['id'] = instance.encode('ascii','ignore')
                    q = np.append(q,added)
                    q = np.sort(q,order='dist')
                else:
                    if float(dist)<float(q[q.shape[0]-1]['dist']):
                        added['dist'] = dist
                        added['id'] = instance.encode('ascii','ignore')
                        q[k-1] = added
                        q = np.sort(q,order='dist')
        #Return max class value within k-NN list
        for class_ in q:
            classes[int(self.classifications[class_[1]].attrs["class"])-1] += 1
        return np.argmax(classes)+1

    def distance(self, data1, data2):
        """Distance measure between data1 and data2"""
        totalDist = 0#abs(data1[0,6]-data2[0,6])#abs(1/data1[0,2]-1/data2[0,2])#-abs(1/data1[0,3]-1/data2[0,3])#-abs(data1.shape[0] - data2.shape[0])
        #return totalDist
        if data2.shape[0] == 0:
            return totalDist

        #Identify similarity of motions
        for movement1 in data1:
            minDist = math.inf
            if math.isfinite(movement1[1]):
                t = np.abs(data2[...,1]-movement1[1])
                t.sort()
                minDist = t[0]
            if math.isfinite(minDist):
                totalDist += minDist

        return totalDist


class Processor(object):
    """Handles the processing and system execution of the program"""
    def fplay(self):
        """
        Tells the renderer to continue rendering at the default speed
        """
        self.frameSkip = 1
        self.pause=False

    def fpause(self):
        """
        tells the renderer to pause the simulation at its current place, 
        the simulation will check every 0.5s if the simulation has been unpaused
        """
        self.frameSkip = 1
        self.pause=True

    def fastForward(self):
        """
        Tells the renderer to fast forward the simulation at 1s per tick, ending if it goes out of the simulation bounds
        should be 20x speed but due to limitations in the display update speed it is limited to this
        """
        self.frameSkip = 20
        self.pause=False

    def rewind(self):
        """
        Tells the renderer to rewind the simulation at 1s per tick, ending if it goes out of the simulation bounds
        should be 20x speed but due to limitations in the display update speed it is limited to this
        """
        self.frameSkip = -20
        self.pause=False

    def fix_pixel(self, x, y, frame, prevFrame, result, confidence, window, **kwargs):
        """
        Applies the following logic applied to each pixel in order to reduce its noise,

        Checks if the pixel in question is inside of the window of interest and if it's current value is zero, this indicates that the pixel is either noise, a flaw in the obtaining method or too close/too far for the sensor to detect it
        Goes out in each direction finding a pixel which is not zero and has a high enough confidence rating to assign itself as a copy of it

        part of the first iteration of noise reduction

        :param x: x coordinate of the pixel within the frame being acted upon
        :param y: y coordinate of the pixel within the frame being acted upon
        :param frame: the frame being processed
        :param prevFrame: the previous frame processed
        :param result: the array which is edited to contain the fixed pixel's value
        :param confidence: an array indicating confidence values for each pixel [OBSOLETE]
        :param window: the size of the search area
        :returns: applies the update rule to the passed in result array
        """
        #self.display(result,0.00001,0,0)
        
        zLimit = 200000
        limit = window          #Sets the window limit within which the algorithm should search for a reliable pixel
        noiseLimit =  1         #The noise limit the algorithm should understand when detecting a pixel
        #Variables indicating area to be fixed, generally should be entire image but can be used to reduce processing time or leave an area deemed as reliable
        xmin = 0                
        xmax = 1000
        ymin = 0
        ymax = 1000
        direction = 0

        for key in kwargs:
            if key=="xmin":
                 ymin=kwargs[key]
            elif key=="xmax":
                ymax=kwargs[key]
            elif key=="ymin":
                xmin=kwargs[key]
            elif key=="ymax": 
                xmax=kwargs[key]

        if frame[x,y]>zLimit or x<xmin or x>xmax or y < ymin or y>ymax:
            result[x,y]=0
            return

        #If true then the pixel must either be: noise, occlusion at an object's edge or a reflective surface
        if frame[x,y] == 0 or prevFrame[x,y] == 0:
            #Due to the time information being kept in the same array as the data the x and y edges are zeroed
            if x==0 or y==0:
                result[x,y]=0
                return

            #Checks if there is an object edge by iterating out in each direction and capturing the first non-zero pixel it encounters, if it's going left or up this pixel must also not have been already set to zero
            found = False
            cx=0    #Right
            _cx=0   #Left
            cy=0    #Down
            _cy=0   #Up

            #Iterate until a suitable pixel is found
            while not found:
                #Increment in all directions
                cx += 1
                _cx -= 1
                cy += 1
                _cy -= 1

                #Check if any of the directions have reached the limit at this distance
                if abs(cx)>limit or abs(_cx)>limit or abs(cy)>limit or abs(_cy)>limit:
                    result[x,y]=0
                    return
                #Ensures that no components over step the array's bounds
                if y+cx>=512:
                    cx-=1
                if y+_cx==0:
                    _cx+=1
                if x+cy>=425:
                    cy-=1
                if x+_cy==0:
                    _cy+=1

                #Checks if a valid pixel value has been found
                if prevFrame[x,y+cx]>0 and frame[x,y+cx]>0 and frame[x,y+cx]<zLimit and confidence[x,y+cx] < noiseLimit:
                    found = True
                    d = int(frame[x,y+cx])
                elif prevFrame[x,y+_cx]>0 and frame[x,y+_cx]>0 and frame[x,y+_cx]<zLimit and confidence[x,y+_cx] < noiseLimit:
                    found = True
                    d = int(frame[x,y+_cx])
                elif prevFrame[x+cy,y]>0 and frame[x+cy,y]>0 and frame[x+cy,y]<zLimit and confidence[x+cy,y] < noiseLimit:
                    found = True
                    d = int(frame[x+cy,y])
                elif prevFrame[x+_cy,y]>0 and frame[x+_cy,y]>0 and frame[x+_cy,y]<zLimit and confidence[x+_cy,y] < noiseLimit:
                    found = True
                    d = int(frame[x+_cy,y])
                elif prevFrame[x+cy,y+cx]>0 and frame[x+cy,y+cx]>0 and frame[x+cy,y+cx]<zLimit and confidence[x+cy,y+cx] < noiseLimit:
                    found = True
                    d = int(frame[x+cy,y+cx])
                elif prevFrame[x+cy,y+_cx]>0 and frame[x+cy,y+_cx]>0 and frame[x+cy,y+_cx]<zLimit and confidence[x+cy,y+_cx] < noiseLimit:
                    found = True
                    d = int(frame[x+cy,y+_cx])
                elif prevFrame[x+_cy,y+cx]>0 and frame[x+_cy,y+cx]>0 and frame[x+_cy,y+cx]<zLimit and confidence[x+_cy,y+cx] < noiseLimit:
                    found = True
                    d = int(frame[x+_cy,y+cx])
                elif prevFrame[x+_cy,y+_cx]>0 and frame[x+_cy,y+_cx]>0 and frame[x+_cy,y+_cx]<zLimit and confidence[x+_cy,y+_cx] < noiseLimit:
                    found = True
                    d = int(frame[x+_cy,y+_cx])
            result[x,y] = d
            return
        
        else:

            result[x,y] = frame[x,y]

    def findConfidence_nozero(self, footage):
        """
        Returns an array indicating confidence values for each of the pixels over a subset of the footage duration

        :param footage: the footage to be analysed, NumPy array format
        :returns: an array with same dimensions as footage containing 1's and 0's indicating pixels rteliabilities
        """
        footage = np.ndarray.astype(footage, np.int16)
        confidence = np.zeros((footage.shape[1], footage.shape[2]))
        
        #Loop over the first 10 frames declaring a pixel unreliable if it is outside of the specified bounds
        for i in range (1,10):
            frameDif = abs(footage[i]-footage[i-1])
            for j in range(footage.shape[1]):
                for k in range(footage.shape[2]):
                    confidence[j,k] = 1 if (frameDif[j,k] > 200 and frameDif[j,k] < 2000)  or frameDif[j,k] > 2600  or confidence[j,k]==1 else 0
        
        return confidence

    def reduce_noise(self, **kwargs): 
        """
        Reduces noise in the selected HDF5 files raw dataset

        :param \**kwargs:
            see below

        :Keyword Arguments:
            * *f1* 
                Indicates that the program is executing independently of user control, the passed value to this parameter indicates the filename to be used when opening the file
            * *classify*
                Given any value tells the program not to update the file variable as it is already set
            * *xmin*
                The minimum x value for cropping
            * *xmax*
                The maximum x value for cropping
            * *ymin*
                The minimum y value for cropping
            * *ymax*
                The maximum y value for cropping
        """

        classify=False
        auto=False

        #Default variable values
        xmin_ = 0
        xmax_ = 600
        ymin_ = 0
        ymax_ = 600
        window = 4

        #Indicates whether to show the execution to the user
        visualise = True

        #Handle kwargs
        for key in kwargs:
            if key=="f1":
                auto=True
            elif key=="classify":
                classify=True
            elif key=="xmin":
                xmin_=kwargs[key]
            elif key=="xmax":
                xmax_=kwargs[key]
            elif key=="ymin":
                ymin_=kwargs[key]
            elif key=="ymax":
                ymax_=kwargs[key]

        #Initialise file variable
        if not classify:
            if not auto:
                #Get file names
                if not self.file.setFile(True):
                    return
            else:
                if not self.file.setFile(True, file=kwargs["f1"]):
                    return

        print("Starting confidence calculation")
        frame = self.file.frameFromFile("raw",1)

        result = np.zeros((425, 512), dtype='int16')

        confidence = np.zeros_like(frame)
        prevFrame = np.copy(frame)
        frame = self.file.frameFromFile("raw", 1)
        
        print("Xmin: "+str(xmin_)+"\nXmax :"+str(xmax_)+"\nYmin :"+str(ymin_)+"\nYmax: "+str(ymax_))

        print("starting noise reduction")

        #Perform nosie reduction while there is a frame to apply it to
        while not frame is None:
            result = np.copy(frame)
            for j in range(1, 425):
                for k in range(0, 512):
                    self.fix_pixel(j, k, frame, prevFrame, result, confidence, window, xmin=xmin_, xmax=xmax_, ymax=ymax_, ymin=ymin_)
            result = result[ymin_:ymax_,xmin_:xmax_]
            
            #Output to file
            result[0,0] = frame[0,0]
            result[0,1] = frame[0,1]
            result[0,2] = frame[0,2]
            result[0,3] = frame[0,3]
            result[0,4] = frame[0,4]
            
            self.file.frameToFile("noise_fix", result, straightCopy=True)
            prevFrame = np.copy(frame)
            frame = self.file.frameFromFile("raw", 1)

            if visualise:
                self.displayFrame(result,0.1,2000,400)
        print("Finishing\r")

    def reduce_noise_2(self):
        """
        Opens a noise fixed footage from an hdf5 file and ouptputs the remedied contents into a new dataset based on which pixels fulfill the confidence requirement
        """

        if not self.file.setFile(True):
            return

        frame = self.file.frameFromFile("noise_fix",1)
        conf = np.zeros((50, frame.shape[0], frame.shape[1]))#(50, 425, 512)
        result = np.zeros([frame.shape[0], frame.shape[1]], dtype='int16')
        for i in range(0,20):
            conf[i] = self.file.frameFromFile("noise_fix", 1)

        self.file.resetCount("noise_fix")

        confidence = self.findConfidence_nozero(conf)
        
        frame = self.file.frameFromFile("noise_fix",1)
        prevFrame = np.copy(frame)

        while not frame is None:
            print(self.file.options["noise_fix_2"])
            #for j in range(1, frame.shape[0]):                    
            #    for k in range(0, frame.shape[1]):
            #        result[j,k] = frame[j,k] if confidence[j,k] == 0 else 0
            result = np.where(np.logical_and(confidence==0,frame >= 1600),prevFrame,frame)
            
            #result = frame.copy()
            #result[:,300:] = 0
            #result[:,:140] = 0

            #Output to file
            result[0,0] = frame[0,0]
            result[0,1] = frame[0,1]
            result[0,2] = frame[0,2]
            result[0,3] = frame[0,3]
            result[0,4] = frame[0,4]

            self.file.frameToFile("noise_fix_2",result,straightCopy=True)

            #self.display(result,1)
            prevFrame = np.copy(frame)
            frame = self.file.frameFromFile("noise_fix",1)
        print("frame is none")

    def review(self):
        """
        Gets a HDF5 file from the user and plays its contents to the user
        """
        self.gui = View()
        thread.start_new_thread(self.gui.initWin,(self,))

        #Gets the file to be read and set it up for reading
        if not self.file.setFile(True):
            return

        pastTime = time.perf_counter()

        frame = self.file.frameFromFile(None, 1)

        while not frame is None:
            t = 0.05 - (time.perf_counter() - pastTime)
            #Ensures wait time is non-negative
            if t <= 0:
                t = 0.0000000000000000001
            pastTime = time.perf_counter()

            #Checks if the GUI thread specified paused playback
            if not self.pause:
                self.displayFrame(frame, t,10000,50)# np.mean(frame)+2*np.std(frame), np.mean(frame)-2*np.std(frame))
                frame = self.file.frameFromFile(None, self.frameSkip)
            else:
                pp.pause(0.5)
        print("Review complete\n")

    def displayFrame(self, frame, pause, max, min):
        """
        Takes in a frame and pushes it to the rendering system then waits 'pause' seconds

        :param frame: 2d NumPy array to display
        :param pause: time in seconds to wait, necessary for MatPlotLib to update ui
        :param max: max value for the cmap
        :param min: min value for the cmap
        """

        #self.gui.updateImg(frame)
        if self.firstCall:
            self.vis = pp.imshow(frame, cmap="jet", vmax=max, vmin=min)
            self.vis.set_data(frame)
            pp.pause(pause)
            pp.draw()
            self.firstCall=False
            if not self.gui is None:
                self.gui.updateTime((str(frame[0,1]),str(frame[0,2]),str(frame[0,3])))
        else:
            self.vis.set_data(frame)
            pp.clim(min,max)
            pp.pause(pause)
            pp.draw()
            if not self.gui is None:
                self.gui.updateTime((str(frame[0,1]),str(frame[0,2]),str(frame[0,3])))

    def getNeighboursBackground(self,x,y,data):
        """
        Finds the non zero neighbours within the window size

        :param x: x of window center
        :param y: y of window center
        :param data: the data upon which the method is applied
        :returns: number of non-zero values in data within window/2 pixels from x,y 
        """

        #Window size
        window = 2
        arr = data[x-window:x+window,y-window:y+window]
        return np.count_nonzero(arr)

    def backgroundEstimation(self, **kwargs):
        """
        Applies the background estimation method to the passed in data, acts as both a pre-processing step for optical flow and generates some general statistics from the data

        :param \**kwargs:
                given below

        :Keyword Arguments:
            * *classify*
                Given any value tells the program not to update the file variable as it is already set
        """
        classify=False
        for key in kwargs:
            if key=="classify":
                classify = True

        if not classify:
            if not self.file.setFile(True):
                return

        #Limits the maximum distance, useful for optical flow when ranging the frame, ensures no anomalous values are incurred
        distLim = 2500

        #Starts GUI options
        self.gui = View()
        thread.start_new_thread(self.gui.initWin,(self,))
        pixelOut = open("pixel.csv",'w')
        frame = self.file.frameFromFile("noise_fix_2", 1)
        frame = np.where(np.logical_or(frame>distLim,frame==0),np.full(frame.shape,distLim),frame)
        oldFrame = frame.copy()

        #Used to generate analysis values from the method's application
        restingDur = 0
        restFreq = 0
        resting = np.array([])
        self.file.deleteAnalysis("movementEst")

        #Creates arrary where dimension 0 indicates whether it is the mean, standard deviation or a binary value indicating whether the pixel frequents as zero or not
        #0 - Old mean
        #1 - New mean
        #2 - Variance
        #3 - Standard deviation
        bgdEst = np.zeros((5,frame.shape[0],frame.shape[1]), 'int32')
        stdDevs = 2
        result = np.zeros_like(bgdEst[0],'int32')

        #Number of frames viewed, used in rolling SD and mean
        n = 0

        #Determine the learning rates
        meanLearningRate = 0.04
        stdDevLearningRate = 0.05

        #Gets a baseline for mean and SD
        print("First")

        bgdEst[0] = frame.copy()

        #Generate initial values for mean and standard deviation over 50 frames, these use standard rolling mean calculations
        for i in range(50):
            n+=1
            print(self.file.options["noise_fix_2"])
            
            bgdEst[1] = bgdEst[0] + (frame-bgdEst[0])/n
            if np.mean(bgdEst[2]) == 0:
                bgdEst[2] = (frame-bgdEst[1])*(frame-bgdEst[0])    
            else:
                bgdEst[2] = bgdEst[2] + (frame-bgdEst[1])*(frame-bgdEst[0])
            bgdEst[3] = np.sqrt(bgdEst[2])

            bgdEst[0] = bgdEst[1]
            
            oldFrame = frame.copy()
            frame = self.file.frameFromFile("noise_fix_2", 1)
            frame = np.where(frame>distLim,np.full(frame.shape,distLim),frame)

        #Update learning rates to reflect that more motion should be encountered
        stdDevLearningRate = 50
        meanLearningRate = 0.05

        out = open("backgroundEst"+str(self.file.getName()[:-5])+".csv",'w')

        print("Second")
        self.file.resetCount("noise_fix_2")

        oldFrame = self.file.frameFromFile("noise_fix_2", 1)
        oldFrame = np.where(oldFrame>distLim,np.full(oldFrame.shape,distLim),oldFrame)
        frame = self.file.frameFromFile("noise_fix_2", 1)

        movement = False
        arr = np.zeros((5), "float32")
        #Movement number, counts how many pixels are different from their statistical average
        mn = 0
        pp.ion()
        movementEst = np.zeros((0,frame.shape[0],frame.shape[1]))
        prev=0
        while not frame is None:
            #Wait for half a second
            if self.pause:
                time.sleep(0.5)
                continue
            n+=1
            
            frame = np.where(np.logical_or(frame>distLim,frame==0),np.full(frame.shape,distLim),frame)
            bgdEst[1] = bgdEst[0] + meanLearningRate*(frame-bgdEst[0])
            bgdEst[2] = bgdEst[2] + 1*np.tanh((frame-bgdEst[1])*(frame-bgdEst[0])/stdDevLearningRate)-0.001*(bgdEst[2]-(frame-bgdEst[1])*(frame-bgdEst[0]))#np.where(t>0,t,np.zeros_like(t))
            bgdEst[3] = np.sqrt(bgdEst[2])
            
            #Resets output variable to prevent any cross-contamination
            result.fill(0)

            result = np.where(bgdEst[1]+stdDevs*bgdEst[3]<frame,frame,0)
            result = np.where(bgdEst[1]-stdDevs*bgdEst[3]>frame,frame,result)
            result = self.backgroundDBSCAN(6,result)
            result = np.where(result>0,result,bgdEst[1])
            bgdEst[0] = bgdEst[1]
            
            dif = np.abs(bgdEst[1] - result)
            print(np.count_nonzero(dif))
            out.write(str(np.count_nonzero(dif))+"\r")
            arr = np.zeros((5), "float32")
            arr[0] = np.count_nonzero(dif)
            self.file.analysisToFile(arr,"movementEst")

            #Copy over time data
            result[0,0:5] = frame[0,0:5]

            #Checks if there is a sufficient amount of movement in the frame to be defined as not being noise
            if np.count_nonzero(dif) > 5:
                movement = True
                mn+=1
                #Update analysis
                if mn > 1:
                    movementEst = np.append(movementEst, [np.abs(prevDif-dif)],0)
                    print("Movement detected")
                prevDif = dif.copy()
                result[0,5] = 1

            #If there was movement but has now ended
            elif movement:
                if movementEst.shape[0] > 2: #Any detected movement for two or less frames is most probably noise
                    #Output to file
                    average = movementEst[np.where(movementEst>0)].size/mn
                    stdDev = np.std(movementEst[np.where(movementEst>0)])
                    
                    arr[1] = stdDev
                    arr[2] = mn
                    arr[3] = np.mean(np.max(movementEst,0))-np.mean(np.min(movementEst,0))
                    arr[4] = self.file.options["noise_fix_2"]-mn
                    self.file.analysisToFile(arr,"movementEst")
                    mn = 0
                    print("Movement finished")
                    movement = False
                    movementEst = np.zeros((0,frame.shape[0],frame.shape[1]))
                    result[0,5] = 1
            else:
                restingDur += 1
                movement = False

            #Code for debugging, allows visualisation of various elements generated above
            #dif = np.abs(bgdEst[1]-frame)
            #self.display(dif,2,dif.max(),dif.min())
            #self.display(bgdEst[1],2,bgdEst[1].max(),bgdEst[1].min())
            #self.display(bgdEst[3],0.00001,bgdEst[3].max(),bgdEst[3].min())
            #self.display(np.abs(result-bgdEst[1]),0.0001,result.max(),0)
            #self.display(result,0.0001,result.max(),0)
            #self.display((bgdEst[1]+stdDevs*bgdEst[3])-result,0.5,2200,500)
            #self.file.frameToFile("bgdest", result, straightCopy=True)
            oldFrame = frame.copy()
            frame = self.file.frameFromFile("noise_fix_2", 1)
        
        #Output to file
        #Ensures value is non-zero
        restFreq += 1
        resting = np.append(resting, restingDur)
        mn = np.mean(resting)
        sd = np.std(resting)
        classes = open("experimentClasses.csv")
        found = False
        file_ = self.file.getName()
        print(file_)
        for line in classes:
            key, value = line.split(',')
            if str.upper(key)==str.upper(file_[:-5]):
                classT = int(value)
                found = True
                break
        classes.close()

        ot = open("backgroundEstClassification.csv",'a')
        ot.write(str(mn)+","+str(sd)+","+str(classT)+",\r")
        ot.close()
        arr = np.zeros((arr.shape), "float32")
        self.file.analysisToFile(arr,"movementEst")

        self.file.setAttribute("file",True,"bgdest")
        out.close()
        pixelOut.close()

    def draw_flow(self, frame, flow, clusters, clusterArr, step=4):
        """
        Original implementation provided by OpenCV documentation, creates an image showing the flows formed in optical flow with their clusters and overlays on the frame data

        :param frame: frame data of current frame, used as background
        :param flow: flows drawn from optical flow
        :param clusters: array of size frame containing values showing flow class membership
        :param clusterArr: clusters centroid locations and centroid predicted locations
        :param step: inverse of flow field displayed density
        :returns: cv2 image which can be displayed and saved
        """

        #CV2 code for extracting flow field display from flow field
        h, w = frame.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis=cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 0, 255))

        #CV2 code for extracting flow field display from flow field using colour indicating direction and brightness indicating magnitude, generally the exact lines are more useful for analysis
        #h, w = flow.shape[:2]
        #fx, fy = flow[:,:,0], flow[:,:,1]
        #ang = np.arctan2(fy, fx) + np.pi
        #v = np.sqrt(fx*fx+fy*fy)
        #hsv = np.zeros((h, w, 3), np.uint8)
        #hsv[...,0] = ang*(180/np.pi/2)
        #hsv[...,1] = 255
        #hsv[...,2] = 2*np.minimum(v*4, 255)
        #vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        #Iterate over clusters plotting vectors with a colour representing class membership, colours are generated from inferno cmap and number of colours generated is made to maximise differentiation between them, mathematically, not visually
        if not clusters is None:
            cmap = pp.get_cmap('inferno')
            colours = cmap(np.linspace(0,1,clusters.max()+3))
            for (x1, y1), (x2, y2) in lines:
                if clusters[y1-2,x1-2]>0:
                    Blue =  0
                    Green = 0
                    Red = 255
                    cv2.circle(vis, (x1, y1),1,colours[int(clusters[y1-2,x1-2])+2]*255, -1)
        #Centroid and prediction colours
        Blue =  255
        Green = 140
        Red = 0
        #Plot centroids and predictions
        if not clusterArr is None:
            for clust in clusterArr:
                if clust.duration>0:
                    cX,cY = clust.getPos()
                    cv2.circle(vis, (cY, cX),6,(Green,Blue,Red), -1)
                    cX,cY = clust.getNextPos()
                    cv2.circle(vis, (int(cY), int(cX)),3,(Red,Blue,Green), -1)
        return vis

    def DBSCAN(self, eps, min_samples, features, minMag):
        """
        Returns a 2-dimensional array of class membership labels for each point
        """
        clusters = np.zeros((features.shape[0],features.shape[1]))
        points = features
        c = 1
        n=0
        n2=0
        t = 0
        for row in points:
            for point in row:
                if point[2] == 0 and clusters[int(point[3]-1),int(point[4]-1)]==0:
                    if point[0] <minMag:
                        point[2] = -1
                    else:
                        point[2] = 1
                        n+=1
                        
                        neighbours = self.getNeighbours(point, points, point[3]-1, point[4]-1, eps, minMag)
                        
                        if len(neighbours)<min_samples:
                            n2+=1
                            clusters[int(point[3]-1),int(point[4]-1)] = -1 #Noise
                        else:
                            c += 1
                            
                            clusters = self.expandCluster(point, points, neighbours, c, eps, min_samples, clusters, minMag)
        return clusters
    
    def expandCluster(self, point, points, neighbours, c, eps, min_samples, clusters, minMag):
        """Expands a DBSCAN cluster
        :param point: coord of point being checked
        :param points: feature space
        :param neighbours: current neighbours to check
        :param c: id of current cluster
        :param eps: closeness requirement
        :param min_samples: minimum number of neighbours required
        :param clusters: cluster space of feature space
        :param minMag: minimum requried magnitude of flow vector
        """
        clusters[int(point[3]-1),int(point[4]-1)] = c
        size = 0
        while neighbours:
            
            neighbour = neighbours.popleft()
            
            #Currently does not belong to a cluster
            if neighbour[2] == 0:
                neighbour[2] = 1
                points[int(neighbour[3]),int(neighbour[4]),2] = 1
                neighbourPts = self.getNeighbours(neighbour, points, neighbour[3]-1, neighbour[4]-1,eps,minMag)
                if len(neighbourPts) >= min_samples:
                   neighbours.extend(neighbourPts)

            
            #Sets the points cluster
            if clusters[int(neighbour[3]-1),int(neighbour[4]-1)] == 0:
                clusters[int(neighbour[3]-1),int(neighbour[4]-1)] = c
                size+=1
            

        if size < self.minSize:
            clusters[np.where(clusters==c)] = 0
        return clusters
    
    def getNeighbours(self, start, points, x, y, eps, minMag):
        """
        Returns a queue of all points in points which are eps or less distance from start
        Starts a point "start" and does a breadth first search of its neighbouring spatial points checking if they are within "eps" distance
        """
        pixelLimit = 4
        q = deque()
        u = points[int(x-pixelLimit):int(x+pixelLimit),int(y-pixelLimit):int(y+pixelLimit)]
        for row in u:
            for point in row:
                dif = start-point
                if point[2] == 0 and point[0]>=minMag and math.sqrt(np.dot(dif[:2],dif[:2]))<eps:
                    q.append(point)
        return q

    def getCentre(self, points, class_):
        """Returns center of a given set of points"""
        x,y=np.where(points==class_)
        
        x = int(np.mean(x))
        y = int(np.mean(y))
        return (x,y)
 
    def getSize(self, points, class_):
        """Gets the size in points of a given cluster"""
        x,y=np.where(points==class_)

        return x.size

    def opticalFlow(self, **kwargs):
        """Applies Farneback optical flow to the file's background estimation result"""
        classify=False
        for key in kwargs:
            if key=="classify":
                classify = True

        if not classify:
            if not self.file.setFile(True):
                return

        name = self.file.getName()[:-5] + '_vid.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        txtOut = open("flow"+str(self.file.getName()[:-5])+".csv",'w')
        frame1 = self.file.frameFromFile("bgdest", 1)
        frame1 = frame1 - np.min(frame1)
        frame1 = np.array(frame1 * float(255/self.file.getFootageMax(None)),"uint8")
        
        out = cv2.VideoWriter(name,fourcc,10,(frame1.shape[1],frame1.shape[0]))
        prvs = cv2.cvtColor(frame1,cv2.COLOR_GRAY2BGR)
        prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
        magWeight = 1
        angWeight = 1
        analysis = np.zeros((7))
        #Features array where index:
        #0, magnitude of vector
        #1, angle of vector
        #2, holder for label which will be 0 when unlabelled, -1 if noise and other positive value indicating it class
        #3, holds the x coordinate of the point
        #4, holds the y coordinate of the point
        features = np.zeros((frame1.shape[0],frame1.shape[1],5))

        features[:,:,3] = np.reshape(np.repeat(np.arange(frame1.shape[0]),frame1.shape[1]),(frame1.shape[0],frame1.shape[1]))
        features[:,:,4] = np.reshape(np.tile(np.arange(frame1.shape[1]),frame1.shape[0]),(frame1.shape[0],frame1.shape[1]))
        flow = None

        clusterInit = True
        clusters = None
        predClusters = None

        clustCount = 0

        frameCounter=0
        while True:
            #Increment frame counter
            print(self.file.options["bgdest"])
            frameCounter += 1

            #Get next frame from file
            frame2 = self.file.frameFromFile("bgdest", 1)

            #If the end of file then exit loop, should be done using an exception
            if frame2 is None:
                    self.file.setAttribute("file",True,"opticalFlow")
                    break
            
            #Check if the frame is in close proximity, temporally, to frames containing movement found in background estimation
            if 1 in self.file.getRange():
                #Setup variable to measure performance
                tim = time.perf_counter()

                #Check that there will not be divide by zero
                if frame2.max() > 0:
                    frame2 = np.array(frame2 * float(255/self.file.getFootageMax(None)),"uint8")
                
                #Necessary for correct format into optical flow function
                next = cv2.cvtColor(frame2,cv2.COLOR_GRAY2BGR)
                next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

                #Perform Farnëback implementation of optical flow
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 2, 15, 1, 20, 1.8, 0)
                
                #Get magnitude and angle arrays of flow field 
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                #Set features array to weighted vectors of mag, ang and zeros 
                features[...,0] = mag * magWeight
                features[...,1] = ang * angWeight
                features[...,2] = np.zeros_like(features[...,2])

                #Sets the minimum size a cluster must be to be valid, should change based on experiment
                self.minSize = 250
                
                #Perform clustering on the flow field
                predClusters = self.DBSCAN(math.pi,4,features,0.4)
                predClusters[np.where(predClusters==0)] = -1
                maxDist = 20

                #Identify clusters for next frame to identify continuous movement
                for cluster in np.unique(predClusters[np.where(predClusters>1)]):
                    cen = self.getCentre(predClusters,cluster)
                    size = self.getSize(predClusters,cluster)                

                    if clusters is None:
                        clusters = np.array([Cluster(cen, size, mag, cluster, predClusters)])
                    elif clusterInit:
                        clusters = np.append(clusters, Cluster(cen, size, mag, cluster, predClusters))
                    else:
                        found = False
                        #Find cluster
                        for clust in clusters:
                            nxtPt = clust.getNextPos()
                            curPt = clust.getPos()
                            if ((predClusters[int(curPt[0]),int(curPt[1])]==cluster) or (nxtPt[0]>0 and nxtPt[1]>0 and nxtPt[0]<predClusters.shape[0] and nxtPt[1]<predClusters.shape[1] and ((clust.getNextPos()[0] < cen[0] + maxDist and clust.getNextPos()[0] > cen[0] - maxDist and clust.getNextPos()[1] < cen[1] + maxDist and clust.getNextPos()[1] > cen[1] - maxDist) or (predClusters[int(nxtPt[0]),int(nxtPt[1])]==cluster)))):
                                clust.update(cen, size, mag, cluster, predClusters)
                                found = True

                        #Create new cluster for flow
                        if not found:
                            clusters = np.append(clusters, Cluster(cen, size, mag, cluster, predClusters))

                #Draw flows and display/write them
                img = self.draw_flow(frame2,flow,predClusters,clusters)
                cv2.imshow('t',img)
                out.write(img)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    self.file.setAttribute("file",True,"opticalFlow")
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png',frame2)
                    cv2.imwrite('opticalhsv.png',bgr)

                #Update points
                prvs = next

                #Delete unused clusters
                deletes = np.array([])
                if not clusters is None:
                    for i in range(clusters.shape[0]):
                        if clusters[i].isNew():
                            clusters[i].age()
                        else:
                            deletes = np.append(deletes, i)
                    analysis = np.zeros((7))

                    #Output results of clusters being deleted
                    for i in deletes:
                        c = clusters[int(i)]
                        velocityArr = np.array([])
                        clustMag = 0
                        if c.duration>4:
                            for frameNum in range(c.accelerationSum.shape[0]):
                                prevMag = np.average(clustMag)
                                clustMag = c.magArray[frameNum][np.where(c.predClustArr[frameNum]==c.clustNumArr[frameNum])]
                                analysis[0] = np.average(clustMag) #c.accelerationSum[frameNum] #Average acceleration, gives a midpoint of the acceleration found at a point
                                analysis[1] = np.average(c.magArray[frameNum][np.where(c.predClustArr[frameNum]==c.clustNumArr[frameNum])]-c.magArray[frameNum-1][np.where(c.predClustArr[frameNum]==c.clustNumArr[frameNum])]) if frameNum>0 else 0 #np.std(c.accelerationSum) #Standard deviation of the acceleration, gives an idea of how uniform the motion was
                                analysis[2] = c.size[frameNum] #Average size of the cluster, gives a good idea of how widespread the uniform motion is
                                analysis[3] = np.average(c.accelerationSum[:frameNum])/frameNum
                                analysis[4] = c.duration #Duration of the clusters life in frames
                                analysis[5] = np.std(c.magArray[frameNum][np.where(c.predClustArr[frameNum]==c.clustNumArr[frameNum])]) #Range of the acceleration
                                analysis[6] = self.file.options["bgdest"]-(c.duration-frameNum)

                                velocityArr = np.append(velocityArr, np.average(clustMag)-prevMag)
                            velocityArr = np.append(velocityArr, 0-np.average(clustMag))
                            
                            acc = 0
                            accSum = 0
                            for j in range(velocityArr.shape[0]):
                                prevAcc = acc
                                acc = velocityArr[j]-prevAcc
                                accSum += math.tanh(abs(acc))
                            accSum += math.tanh(abs(0-acc))

                            avAcc = accSum/c.accelerationSum.shape[0]

                            txtOut.write(str(avAcc)+","+str(c.accelerationSum.shape[0]))
                            txtOut.write("\r")
                            analysis[0] = avAcc
                            analysis[1] = c.accelerationSum.shape[0]
                            self.file.analysisToFile(analysis,"opticalFlow")    

                    #Remove clusters
                    for i in deletes:
                        clusters = np.delete(clusters, i)
                    clusterInit = False

        #Ensure analysis is non-empty
        analysis.fill(0)
        self.file.analysisToFile(analysis,"opticalFlow")   

        #Release resources 
        out.release()  
        txtOut.close()

    def trim(self, **kwargs):
        """Trims the footage to the predefined locations"""
        classify=False
        for key in kwargs:
            if key=="classify":
                classify = True

        if not classify:
            if not self.file.setFile(True):
                return
        
        duration = 100

        end = self.file.getFrameLength(None) - 100 #Trim last 5 seconds of footage
        start = end - (duration*20) #Defines duration, 20fps for duration seconds

        self.file.trim(None, start, end)

    def classify(self, file_):
        """Run methods on a single instance then classify all"""
        self.file = Hdf5FileHandler(self)
        self.pause = False
        self.frameSkip = 1
        self.gui = None
        self.firstCall=True
        print("Classifying: "+file_)
        classes = open("experimentClasses.csv")
        found = False
        for line in classes:
            key, value = line.split(',')
            if str.upper(key)==str.upper(file_[:-5]):
                classT = int(value)
                found = True
                break


        if not self.file.f is None:
            self.file.close()
        self.file.setFile(True, file = file_)
            
        print("Class: "+str(classT)+"\r")     

        #Apply methods
        self.backgroundEstimation(classify="True")
        self.file.resetCount("bgdest")
        print("Applying optical flow, Farneback")
        self.file.deleteAnalysis("opticalFlow")
        self.opticalFlow(classify="True")
        print("Outputing analysis")
        analysisMovEst = self.file.analysisFromFile("movementEst")
        analysisOpticalFlow = self.file.analysisFromFile("opticalFlow")
        self.outputToDataset(classT,analysisMovEst,analysisOpticalFlow,analysisOpticalFlow.shape[0])
            
        classes.close()

    def classifySingle(self):
        """Run methods on all instances found then classifies all"""
        for file_ in glob.glob("*.hdf5"):
            if file_ == "classifications.hdf5":  
                continue
            print("Classifying: "+file_)
            classes = open("experimentClasses.csv")
            found = False
            for line in classes:
                key, value = line.split(',')
                if str.upper(key)==str.upper(file_[:-5]):
                    classT = int(value)
                    found = True
                    break

            if not found:
                classes.close()
                continue

            if not self.file.f is None:
                self.file.close()
            self.file.setFile(True, file = file_)
            
            print("Class: "+str(classT)+"\r")

            #Apply methods, assumes noise fix and trimming has already occurred
            self.backgroundEstimation(classify="True")
            self.file.resetCount("bgdest")
            print("Applying optical flow, Farneback")
            self.file.deleteAnalysis("opticalFlow")
            self.opticalFlow(classify="True")
            print("Outputing analysis")
            analysisMovEst = self.file.analysisFromFile("movementEst")
            print("moveibir")
            print(analysisMovEst)
            analysisOpticalFlow = self.file.analysisFromFile("opticalFlow")
            self.outputToDataset(classT,analysisMovEst,analysisOpticalFlow,analysisOpticalFlow.shape[0])
            
            classes.close()

    def outputToDataset(self,classT,movement,opticalFlow,clusterCount):
        """Output analysis results to the relevant dataset"""
        #Make a prediction
        classifications = h5py.File('classifications.hdf5','a')

        if str(self.file.getName()[:-5]) in classifications:
           del classifications[str(self.file.getName()[:-5])] 
        dataset = classifications.create_dataset(str(self.file.getName()[:-5]), (opticalFlow.shape[0],4), maxshape=(None, None), chunks=True)
    
        #Get movement estimate from background estimation application and generate average acceleration
        velArr = np.zeros(movement.shape[0]+1)
        print(movement)
        velArr[0]=movement[0,0]

        #Change in movement amount
        for i in range(1,movement.shape[0]-1):
            velArr[i+1] = movement[i+1,0]-movement[i,0]
        velArr[movement.shape[0]] = -movement[movement.shape[0]-1,0]

        sum = abs(velArr[0])

        #Change in movement speed
        for i in range(velArr.shape[0]-1):
            sum += abs(velArr[i+1]-velArr[i])
        sum += -velArr[velArr.shape[0]-1]

        complexity = sum/movement.shape[0]
    
        #Ouput to results file
        dataset[:,0:2] = opticalFlow[:,0:2]
        dataset[:,2] = np.mean(movement[...,0])/movement.shape[0]
        dataset[:,3] = complexity
        dataset.attrs["class"] = classT
                
        classifications.close()

    def write(self):
        """
        Takes in frames from an active kinect sensor and saves them to a .hdf5 file specified by the user
        """

        self.file.setFile(False)
        recording = True
        frameCount = 0

        print("\nStarting recording...\n")

        #Sets timers for counting frames and checking for bottle necks
        timeSinceLastFrame = time.perf_counter()
        timeLastSecond = time.perf_counter()
                
        #Keep looping until ctrl+c is pressed, would be good to implement a timeout
        try:
            while recording:
                if time.perf_counter() - timeLastSecond >= 1:
                        numberOfFramesInSecond = 0
                        timeLastSecond = time.perf_counter()

                if self._kinect.has_new_depth_frame():                    

                    frameCount += 1
                    frame = self._kinect.get_last_depth_frame()
                    frame = np.reshape(frame,(424,512))

                    self.file.frameToFile("raw",frame)

                    #Checks if the program should wait to keep consistent FPS, should do this unless system can't keep up
                    if time.perf_counter()-timeSinceLastFrame<1/self.file.fps:
                        time.sleep(1/self.file.fps-(time.perf_counter()-timeSinceLastFrame))
                    else:
                        print("System hanging, check no IO bottle neck")

                    timeSinceLastFrame = time.perf_counter()

                    frame = None
                else:
                    self.file.setAttribute(None, frameCount, length)
                    recording = False
        #Exit when Ctrl+C is pressed
        except KeyboardInterrupt:
            pass

        self.file.setAttribute("file",False,"noise_fix")
        self.file.setAttribute("file",False,"bgdest")
        self.file.setAttribute("file",False,"opticalFlow")

        # Close our Kinect sensor and file
        self.file.close()

        self._kinect.close()
        print("System closing, releasing resources")

    def convert(self, **kwargs):
        """
        Converts a sequence of .kin files into their .hdf5 counterparts using lzf compression and chunking
        """
        auto=False
        for key in kwargs:
            if key=="f1":
                auto=True
                olddir = kwargs["f1"]
        if not auto:
            #Get file names
            print("Please enter old directory name")
            olddir = input()
        
            if not self.file.setFile(False):
                return
        else:
            if not self.file.setFile(False, file=kwargs["f2"]):
                return

        filenum = 0

        #Loop through files and copy data into hdf5 file
        while os.path.isfile(olddir+"\data"+str(filenum)+".kin"):
            #Open file and retrieve data from it
            oldfile = open(olddir+"\data"+str(filenum)+".kin", 'rb', 512)
            filenum = filenum+1
            footage = np.fromfile(oldfile, dtype='uint16')

            #Print to user initial time of recording and length
            print("Start time: "+str(footage[0])+":"+str(footage[1])+"."+str(footage[2])+"."+str(footage[3])+"\n Warnings:")
            print("Total duration in seconds: "+str((1/footage[4])*(footage.size/217600)))
        
            #Reshape the single dimension array into an a series of 2 dimensional arrays
            #If an error is thrown here then the daat is corrupted or in the wrong format
            print(footage.size/217600)
            footage = np.reshape(footage, (int(footage.size/217600), 425, 512))

            #Iterate over all frames
            for i in range(footage.shape[0]):
                print(i)
                self.file.frameToFile("raw", footage[i], straightCopy=True)

        self.file.setAttribute("file",False,"noise_fix")
        self.file.setAttribute("file",False,"bgdest")
        self.file.setAttribute("file",False,"opticalFlow")

    def run(self):
        """
        Initialises parameters and displays the main menu to the user
        """
        # Kinect runtime object, we want only color and body frames, commented due to development computer not having required library installed and would not run, works when correct libraries are installed
        #self._kinect = PyKinectRuntime.PyKinectRuntime(8)
        self.pause = False
        self.frameSkip = 1
        self.gui = None

        choice = 1
        while choice > 0:
            #Creates a clean instance of the file handler for each user request
            self.file = Hdf5FileHandler(self)
            #Variable used for display purposes
            self.firstCall = True
            print("Please select an option: \n1.Playback a file\n2.Record a file\n3.Trim a files duration\n4.Apply noise reduction to chosen file's raw footage\n5.Apply background estimation to noise fixed footage\n6.Apply optical flow to background "+
                    "estimated footage\n7.Optimise file size\n8.Perform classification test\n9.Convert a file from the old file format to hdf5\n10.Classify all experiments sequentially\n11.Convert a file using the old storage solution to the new solution")
            choice = input()
            if choice == '1':
                self.review()
                choice = 1
            elif choice == '2':
                self.write()
                choice = 1
            elif choice == '3':
                self.trim()
                choice=1
            elif choice == '4':
                self.reduce_noise()
                choice = 1
            elif choice == '5':
                self.backgroundEstimation()
                choice = 1
            elif choice == '6':
                self.opticalFlow()
                choice = 1
            elif choice == '7':
                choice = 1
                self.file.setFile(True)
                file2 = Hdf5FileHandler(self)
                file2.setFile(True)
                self.file.transfer(file2)
                file2.close()
            elif choice == '8':
                Classifier()
                choice = 1
            elif choice == '9':
                self.convert()
                choice = 1
            elif choice == '10':
                self.classifySingle()
                choice = 1
            elif choice == '11':
                self.file.setFile(True)
                file2 = Hdf5FileHandler(self)
                file2.setFile(True)
                self.file.convert(file2)
                choice = 1

            else:
                choice = -1
            
            if not self.file.f is None:
                self.file.close()

    def __init__(self):
        self.done = False

#Start the program
execution = Processor();
execution.run();