# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:43:19 2020

@author: Alexander
"""
from settings import classes, cur_path
import numpy as np
import os
from PIL import Image
import pandas as pd


def convertOneHot(oldTarget, n_classes=classes):
    newTarget = np.zeros((oldTarget.shape[0], n_classes))
    for item in range(0, oldTarget.shape[0]):
        newTarget[item][oldTarget[item]] = 1

    return newTarget

#Shuffles the oldData, with the targets
def shuffle(oldData, oldTarget):
    np.random.seed(421)
    randIndx = np.arange(len(oldData))
    np.random.shuffle(randIndx)
    newData, newTarget = oldData[randIndx], oldTarget[randIndx]
    return newData, newTarget


# Data set is 39,210 datapoints (76%)
def loadData(n_classes=classes):
    data=[]
    targets=[]

    #Load train data
    print("Starting to load images")
    #
    for i in range(n_classes):
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

# Train data: (31368, 30, 30, 3)
# Validation data: (7841, 30, 30, 3)
def loadSplitTrainValidation(n_classes=classes):
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
    
    testData=[]
    testTarget=[]
    
    
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
    
    #size of 1 image
    dimw = 30*30*3
    num_examples = testTarget.shape[0]

    X = np.zeros((num_examples,dimw))

    for i in range(num_examples):
        X[i]=testData[i].flatten()
        
    return X, testTarget