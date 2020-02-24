# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:38:24 2020

@author: Alexander
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:02:33 2020
@author: AlexanderApostolov
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
classes=43
cur_path=os.getcwd()


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
    for i in range(n_classes):
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
                
    data=np.array(data)
    targets=np.array(targets)
    
    
    size = data.shape[0]
    print("Train&Validation Data loaded - total datapoints: ", size)


    return data, targets

# Train data: (31368, 30, 30, 3)
# Validation data: (7841, 30, 30, 3)
def loadSplitTrainValidation(n_classes=classes):
    data, targets = loadData()
    num_examples = data.shape[0]
    
    #size of 1 image
    dimw = 30*30*3

    X = np.zeros((num_examples,dimw))

    for i in range(0,num_examples):
        X[i]=data[i].flatten()
    
    #Take 20% for validation data
    sizeValidation = num_examples//5
    sizeTrain = num_examples - sizeValidation
    
    print("    Train data: ", sizeTrain)
    print("    Validation data: ", sizeValidation)    
    
    #Randomly take 20% of the data
    X, targets = shuffle(X, targets)
    
    trainData = X[:sizeTrain]
    trainTarget = targets[:sizeTrain]
#    trainData = X[:2000]
#    trainTarget = targets[:2000]
    trainTarget = convertOneHot(trainTarget, n_classes)
    
    validationData = X[sizeTrain:]
    validationTarget = targets[sizeTrain:]
#    validationData = X[2000:2400]
#    validationTarget = targets[2000:2400]
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

def cnnTF(keep1=0.25, keep2=0.5, learning_rate=1e-2, batch_size=64, epochs=15, onlyFinal=False, n_classes=classes):
    tf.set_random_seed(421)
    #Xavier initializer
    initializer = tf.contrib.layers.xavier_initializer() 

    wc= tf.Variable(initializer([3,3,3,32]))
    bc =tf.Variable(initializer([32]))
    #zero-mean Gaussians initialization with variance 2/unitsin+unitsout
    sig1 = np.sqrt(2/((15*15*32)+784))
    sig2 = np.sqrt(2/(784+10)) 
    w1 = tf.Variable(tf.random_normal([15*15*32, 784], stddev = sig1), name='w1')
    b1 = tf.Variable(tf.random_normal([784] ,stddev = sig1), name = 'b1')
    
    w2 = tf.Variable(tf.random_normal([784, 43],stddev = sig2), name='w2')
    b2 = tf.Variable(tf.random_normal([43], stddev = sig2),  name = 'b2')
    
    
    x = tf.placeholder(tf.float32, shape=(None, 30*30*3), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 43), name='y')
    
    #1. Input Layer
    input_layer = tf.reshape(x, shape=[-1, 30, 30, 3])
    
    #2. Convolutional layer
    conv = tf.nn.conv2d(input=input_layer, filter=wc, strides=[1,1,1,1], padding="SAME")
    conv = tf.nn.bias_add(conv, bc)
    
    #3. Relu activation
    relu1 = tf.nn.relu(conv)
    
    #4. Batch normalization layer 
    
    batch_mean, batch_var = tf.nn.moments(relu1,[0])
    normal = tf.nn.batch_normalization(relu1, batch_mean, batch_var, offset = None, scale = None, variance_epsilon = 1e-3)
    
    #5 A 2  2 max pooling layer
    
    maxpool = tf.nn.max_pool(normal, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    #6 Flatten layer 
    
    flat = tf.reshape(maxpool, [-1, 15*15*32])
    
    #7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
    
    full = tf.add(tf.matmul(flat, w1), b1)
    
    #8.Dropout if needed + ReLU activation

    drop_layer = tf.nn.dropout(full, keep_prob=0.25)
    relu2 = tf.nn.relu(drop_layer)
        
    #9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
    
    out = tf.add(tf.matmul(relu2, w2), b2)
    
    #10. Softmax output
    
    softmax_layer = tf.nn.softmax(out)
    
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    correct_pred = tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
     
    init = tf.global_variables_initializer()
        
    #prepareTables
    trainLoss = np.zeros(epochs)
    trainAccuracy = np.zeros(epochs)
    validationLoss = np.zeros(epochs)
    validationAccuracy = np.zeros(epochs)

    #initialize the datasets
    X, trainTargets,Xvalid,validationTargets=loadSplitTrainValidation()
    
    print("Data loaded")
    
    saver=tf.train.Saver(max_to_keep=3)    
    
    with tf.Session() as sess:
        sess.run(init)
        print("Variables initialized")
        num_examples=trainTargets.shape[0]
        number_of_batches = num_examples//batch_size
    
        for step in range(epochs):
            #Shuffle after each epcoh
            flat_x_shuffled,trainingLabels_shuffled = shuffle(X, trainTargets)
            print("----------------------------------------")
            print("Epoch ", (step+1))
            
            
            for minibatch_index in range(number_of_batches):
                print('\rMinibatch {}/{}'.format(minibatch_index, number_of_batches), end='\r')
                
                #select miniatch and run optimizer
                minibatch_x = flat_x_shuffled[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]
                minibatch_y = trainingLabels_shuffled[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]           
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            if(step==epochs-1 or not(onlyFinal)):
#                lossTrain, accTrain = sess.run([loss_op, accuracy], feed_dict={x: flat_x_shuffled, y: trainingLabels_shuffled})
                XvalidShuffled, validationTargetsShuffled = shuffle(Xvalid, validationTargets)
                XvalidSubset, validationTargetsSubste = XvalidShuffled[:640], validationTargetsShuffled[:640]
                lossValid, accValid = sess.run([loss_op, accuracy], feed_dict={x: XvalidSubset, y: validationTargetsSubste})

#                trainLoss[step]=lossTrain
#                trainAccuracy[step]=accTrain
                validationLoss[step]=lossValid
                validationAccuracy[step]=accValid

#                print("Epoch " + str(step+1) + ", Train Loss= " + \
#                              "{:.8f}".format(lossTrain) + ", Training Accuracy= " + \
#                              "{:.8f}".format(accTrain) + "\n") + \
#                              "Validation Loss=" +\
#                              "{:.8f}".format(lossValid) + ", Validation Accuracy= " + \
#                              "{:.8f}".format(accValid) + "\n")
                
                print("Epoch " + str(step+1) + ", Validation Loss=" +\
                              "{:.8f}".format(lossValid) + ", Validation Accuracy= " + \
                              "{:.8f}".format(accValid) + "\n")
                
            else:
                print("Epoch " + str(step+1) + " (no data, only final losses and accuracies will be saved)")
            
            if(step==0):
                save_path=saver.save(sess, './savedModels/modelTrafficSigns', global_step=step)
                print("Model saved in path: %s" % save_path)
            else:
                #no need to save the graph
                save_path=saver.save(sess, './savedModels/modelTrafficSigns', write_meta_graph=False, global_step=step)
                print("Model saved in path: %s" % save_path)
                
        print("Optimization Finished!")

    return trainLoss, trainAccuracy, validationLoss, validationAccuracy
    
def main():
    trainLoss, trainAccuracy, validationLoss, validationAccuracy = cnnTF()
    with open('optTrafficSigns.pkl', 'wb') as f: 
        pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy], f)
    return

main()
    
#sess=tf.Session()    
##First let's load meta graph and restore weights
#saver = tf.train.import_meta_graph('/tmp/modelTrafficSigns.ckpt')
#saver.restore(sess,tf.train.latest_checkpoint('./'))

def cnnKeras():
    trainData, trainTargets,_,_=loadSplitTrainValidation()
    model=Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=trainData.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return

def testLoadedData():
    trainData, trainTargets,validationData,validationTargets=loadSplitTrainValidation()

    img = Image.fromarray(trainData[456], 'RGB')
    img.show()
    print(trainTargets[456])
    img = Image.fromarray(trainData[1234], 'RGB')
    img.show()
    print(trainTargets[1234])
    img = Image.fromarray(trainData[3333], 'RGB')
    img.show()
    print(trainTargets[3333])
    img = Image.fromarray(trainData[5555], 'RGB')
    img.show()
    print(trainTargets[5555])
    return


#cnnKeras()
#loadTrainValidationData()