# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:43:19 2020

@author: Alexander

This script is used to load and prepare the datasets and to find saved metafiles

This script requires numpy, PIL and pandas
"""
from settings import classes, cur_path
import numpy as np
import os
from PIL import Image
import pandas as pd
from os import listdir
from os.path import isfile, join


def convertOneHot(oldTarget, n_classes=classes):
    """Converts an array of classes(integers) into one-hot encoding
       
    Parameters
    ----------
    oldTarget : numpy array
        an array of int classes
    n_classes : int, optional
        total number of classes (default is settings.classes)
        
    Returns
    -------
    newTarget
        2 dimensional array with one hot encoded classes
    """    
    
    newTarget = np.zeros((oldTarget.shape[0], n_classes))
    for item in range(0, oldTarget.shape[0]):
        newTarget[item][oldTarget[item]] = 1

    return newTarget

def shuffle(oldData, oldLabels):
    """Returns an shuffled copy of the data and their labels
       
    Parameters
    ----------
    oldData : numpy array
        array containg the data
    oldTarget : numpy array
        array containing the labels
        
    Returns
    -------
    newData : numpy array
        shuffled copy of the data
    newTarget : numpy array
        shuffled copy of the targets
    """    

    np.random.seed(421)
    randIndx = np.arange(len(oldData))
    np.random.shuffle(randIndx)
    newData, newTarget = oldData[randIndx], oldLabels[randIndx]
    return newData, newTarget


def loadData(n_classes=classes):
    """Loads the train dataset with its labels
       
    The Data set is 39,210 datapoints (76% of all datapoints)

    
    Parameters
    ----------
    n_classes : int, optional
        total number of classes (default is settings.classes)
        
    Returns
    -------
    data : numpy array
        all datapoints representing the traffic sign images
    targets  : numpy array
        targets of the data
    """  
    
    data=[]
    targets=[]

    print("Starting to load images")
    
    for i in range(n_classes):
        #updates the state of loading
        print('\rLoading class: {}/{}'.format(i, n_classes), end='\r')
        path = os.path.join(cur_path, 'train', str(i))
        
        #Return a list containing the names of the files in the directory.
        images = os.listdir(path)
        
        for curr in images:
            try:
                photo = Image.open(path+'\\'+curr)
                photo = photo.resize((30,30))
                photo = np.array(photo)
                data.append(photo)
                targets.append(i)
            except:
                print("Error while loading train image")
    print('\rLoading class: {}/{}'.format(n_classes, n_classes), end='\n')           
    data=np.array(data)
    targets=np.array(targets)
    
    
    size = data.shape[0]
    print("Train&Validation Data loaded - total datapoints: ", size)


    return data, targets


def loadSplitTrainValidation(n_classes=classes, quickLoadForTest=False):
    """Loads the train dataset with its labels and splits it into Train and Validation
       
    The Data set is 39,210 datapoints
    - Train dataset is 80% of this 
    - Validation dataset is 20% of this
    
    Train data: (31368, 30, 30, 3)
    Validation data: (7841, 30, 30, 3)
    
    Parameters
    ----------
    n_classes : int, optional
        total number of classes (default is settings.classes)
    quickLoadForTest : bool, optional
        for debugging, only loads two classes (default is False)
        
    Returns
    -------
    trainData : numpy array
        images used for training
    trainTarget  : numpy array
        targets of the train images
    validationData : numpy array
        images used for validation
    validationTarget  : numpy array
        targets of the validation images
    """  
    
    if quickLoadForTest:
        print("Quick Loading only two classes (use only for debugging)")
        X, targets = loadData(n_classes=2)
    else:
        X, targets = loadData()
    num_examples = X.shape[0]

    #Take 20% for validation data
    sizeValidation = num_examples//5
    sizeTrain = num_examples - sizeValidation
    
    print("    Train data: ", sizeTrain)
    print("    Validation data: ", sizeValidation)    
    
    #Randomly take 20% of the data
    X, targets = shuffle(X, targets)
    
    trainData = X[:sizeTrain]
    trainTarget = targets[:sizeTrain]
    trainTarget = convertOneHot(trainTarget, n_classes)
    
    validationData = X[sizeTrain:]
    validationTarget = targets[sizeTrain:]
    validationTarget = convertOneHot(validationTarget, n_classes)

    return trainData, trainTarget, validationData, validationTarget


# Test  data set is 12'631 datapoints (24%)
def loadTestData(n_classes=classes):
    """Loads the test dataset with its labels
       
    The Data set is 12,631 datapoints (24% of all datapoints)
    
    Parameters
    ----------
    n_classes : int, optional
        total number of classes (default is settings.classes)
        
    Returns
    -------
    testData : numpy array
        datapoints representing the traffic sign images
    testTarget  : numpy array
        targets of the data
    """  
    
    testData=[]
    testTarget=[]
    
    print("Loading test data...")
    test_file = pd.read_csv('Test.csv')
    testTarget=test_file["ClassId"].values
    images=test_file["Path"].values
    
    for curr in images:
        try:
            photo = Image.open(curr)
            photo = photo.resize((30,30))
            testData.append(np.array(photo))
        except:
            print("Error while loading test image")
    
    testData = np.array(testData)
    testTarget = np.array(testTarget)
    testTarget = convertOneHot(testTarget, n_classes)
    
    print("Test data loaded")
    return testData, testTarget

def findLatestMetaFile(name):
    """Finds the path of the latest .meta file
       
    The path of the meta file is of the form:
    './savedModels/'+run_name+"/state_at_step"
    
    Parameters
    ----------
    name : str
        name of the model
        
    Returns
    -------
    biggest_step : int
        step that was reached last during traing, -1 if model not found
    file_with_biggest_step  : str
        path to the .meta file
    """  

    directory = "./savedModels/"+name
    if not(os.path.isdir(directory)):
        print("Meta file not found (directory not found)")
        return -1, ""

    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    biggest_step=-1
    file_with_biggest_step=""
    for file in onlyfiles:
        filename, file_extension = os.path.splitext(file)
        beginning = "state_at_step-"
        if file_extension==".meta" and filename.startswith(beginning):
            rest=filename[len(beginning):]
            try:
                int_value = int(rest)
                if int_value > biggest_step:
                    biggest_step=int_value
                    file_with_biggest_step=filename+file_extension
            except ValueError:
                pass
    if biggest_step!=-1:
        print("Biggest step found is ", biggest_step)
        print("Meta file is " + file_with_biggest_step)
    else:
        print("Meta file not found")
    return biggest_step, file_with_biggest_step