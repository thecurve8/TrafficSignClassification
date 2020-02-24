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
    data, targets = loadData()
    num_examples = data.shape[0]
    
    #size of 1 image
    dimw = 30*30*3

    X=data
#    X = np.zeros((num_examples,dimw))
#
#    for i in range(0,num_examples):
#        X[i]=data[i].flatten()
    
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

def cnnTF(keep1=0.25, keep2=0.5, learning_rate=1, batch_size=64, epochs=15, onlyFinal=False, n_classes=classes):
    tf.set_random_seed(421)
    #Xavier initializer
    initializer = tf.contrib.layers.xavier_initializer() 

    w1= tf.Variable(initializer([5,5,3,32]), name='w1')
    b1 =tf.Variable(initializer([32]), name='b1')
    
    w2 = tf.Variable(initializer([5,5,32,32]), name='w2')
    b2 = tf.Variable(initializer([32]), name = 'b2')
    
    w3= tf.Variable(initializer([3,3,32,64]), name='w3')
    b3 =tf.Variable(initializer([64]), name='b3')
    
    w4 = tf.Variable(initializer([3,3,64,64]), name='w4')
    b4 = tf.Variable(initializer([64]), name = 'b4')
    
    w5 = tf.Variable(initializer([3*3*64,256]), name='w5')
    b5 = tf.Variable(initializer([256]), name = 'b5')
    
    w6 = tf.Variable(initializer([256,n_classes]), name='w6')
    b6 = tf.Variable(initializer([n_classes]), name = 'b6')
    
    
    x = tf.placeholder(tf.float32, shape=(None, 30,30,3), name='x')
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name='y')
    
    #1. Input Layer
    input_layer = tf.reshape(x, shape=[-1, 30, 30, 3])
    
    #2. Convolutional layer 1
    layer2 = tf.nn.conv2d(input=input_layer, filter=w1, strides=[1,1,1,1], padding="VALID")
    layer2 = tf.nn.bias_add(layer2, b1)
    
    #3. Relu activation
    layer3 = tf.nn.relu(layer2)
    
    #4. Convolutional layer 2
    layer4 = tf.nn.conv2d(input=layer3, filter=w2, strides=[1,1,1,1], padding="VALID")
    layer4 = tf.nn.bias_add(layer4, b2)
    
    #5. Relu activation
    layer5 = tf.nn.relu(layer4)
    
    #6 A 2x2 max pooling layer 
    layer6 = tf.nn.max_pool(layer5, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID") 
    
    #7 dropout layer
    layer7 = tf.nn.dropout(layer6, keep_prob=keep1)
    
    #8. Convolutional layer 3
    layer8 = tf.nn.conv2d(input=layer7, filter=w3, strides=[1,1,1,1], padding="VALID")
    layer8 = tf.nn.bias_add(layer8, b3)
    
    #9. Relu activation
    layer9 = tf.nn.relu(layer8)
    
    #10. Convolutional layer 4
    layer10 = tf.nn.conv2d(input=layer9, filter=w4, strides=[1,1,1,1], padding="VALID")
    layer10 = tf.nn.bias_add(layer10, b4)
    
    #11. Relu activation
    layer11 = tf.nn.relu(layer10)
    
    #12 A 2x2 max pooling layer 
    layer12 = tf.nn.max_pool(layer11, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID") 
    
    #13 dropout layer
    layer13 = tf.nn.dropout(layer12, keep_prob=keep1)    
    
    #14 Flatten layer 
    layer14 = tf.reshape(layer13, [-1, 3*3*64])
    
    #15. Fully connected layer (with 256 output units)
    layer15 = tf.add(tf.matmul(layer14, w5), b5)
    
    #16. Relu activation
    layer16 = tf.nn.relu(layer15)
    
    #17 dropout layer
    layer17 = tf.nn.dropout(layer16, keep_prob=keep2)    
    
    #18. Fully connected layer (with 256 output units)
    out = tf.add(tf.matmul(layer17, w6), b6)
    
    #19. SoftMax activation
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
    
#    print(trainTargets[0])
#    print(trainTargets[1])
#    print(trainTargets[2])
#    
#    img = Image.fromarray(X[0], 'RGB')
#    img.show()
#    img = Image.fromarray(X[1], 'RGB')
#    img.show()
#    img = Image.fromarray(X[2], 'RGB')
#    img.show()
    
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
                XvalidSubset, validationTargetsSubstet = XvalidShuffled[:640], validationTargetsShuffled[:640]
                lossValid, accValid, sm_layer, = sess.run([loss_op, accuracy, softmax_layer], feed_dict={x: XvalidSubset, y: validationTargetsSubstet})
                print(sm_layer[:3])
                print(validationTargetsSubstet[:3])
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

#main()
    
#sess=tf.Session()    
##First let's load meta graph and restore weights
#saver = tf.train.import_meta_graph('/tmp/modelTrafficSigns.ckpt')
#saver.restore(sess,tf.train.latest_checkpoint('./'))

def cnnKeras():
    trainData, trainTargets,vData,vTargets=loadSplitTrainValidation()
    model=Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=trainData.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    epochs=15
    history = model.fit(trainData, trainTargets, batch_size=64, epochs=epochs, validation_data=(vData, vTargets))
    return
cnnKeras()
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