# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:02:33 2020
@author: AlexanderApostolov

This script is used to train models for traffic sign recognition

During training loss and accuracy are measured

To view logged values of accuracy and loss
>>>tensorboard --logdir="./logs" --port 6006
To see results go to http://localhost:6006/#scalars

This script requires tensorflow, numpy, pickle, matplotlib, PIL
"""
import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from settings import classes, cur_path
from dataLoadingAndManipulation import shuffle, loadSplitTrainValidation, findLatestMetaFile
from layersNN import conv2d, max_pooling2d, dropout, flatten, dense
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def cnnTF(run_name, batch_size=64, epochs_to_run=15, onlyFinalValidationStats=False, n_classes=classes, keep1_prob=0.75, keep2_prob=0.5, learning_rate=0.001):
    """Trains a new model for traffic sign recognition
    
    Parameters
    ----------
    run_name : str
        name of the model
    batch_size : int, optional
        size of batches used for training (default is 64)    
    epochs_to_run : int, optional
        epochs to run in the training (default is 15)    
    onlyFinalValidationStats : bool, optional
        takes validation loss and accuracy only at the end of training
        used to save time (default is False)    
    n_classes : int, optional
        total number of classes (default is settings.classes)    
    keep1_prob : float, optional
        keep probability of the first dropout layer (default is 0.75)    
    keep2_prob : float, optional
        keep probability of the second dropout layer (default is 0.5)    
    learning_rate : float, optional
        learning rate of the adam optimizer (default is 0.001)    
        
    Returns
    -------
    trainLoss : numpy array
        loss on the train batches throughout training, recorded after each minibatch
    trainAccuracy  : numpy array
        accuracy on the train batches throughout training, recorded after each minibatch
    validationLoss : numpy array
        loss on the validation dataset throughout training, recorded after each epoch
    validationAccuracy  : numpy array
        accuracy on the validation dataset throughout training, recorded after each epoch
    """  
        
    # delete the current graph
    tf.reset_default_graph()
    
    tf.set_random_seed(421)
    global_step = tf.train.get_or_create_global_step()

    #Initializers
    xavierInitializer = tf.contrib.layers.xavier_initializer(uniform=True) 
    zeroInitializer = tf.zeros_initializer()
    
    #Weights and biases
    w1= tf.Variable(xavierInitializer([5,5,3,32]), name='w1')
    b1 =tf.Variable(zeroInitializer([32]), name='b1') 
    w2 = tf.Variable(xavierInitializer([5,5,32,32]), name='w2')
    b2 = tf.Variable(zeroInitializer([32]), name = 'b2')    
    w3= tf.Variable(xavierInitializer([3,3,32,64]), name='w3')
    b3 =tf.Variable(zeroInitializer([64]), name='b3')    
    w4 = tf.Variable(xavierInitializer([3,3,64,64]), name='w4')
    b4 = tf.Variable(zeroInitializer([64]), name = 'b4')    
    w5 = tf.Variable(xavierInitializer([3*3*64,256]), name='w5')
    b5 = tf.Variable(zeroInitializer([256]), name = 'b5')    
    w6 = tf.Variable(xavierInitializer([256,n_classes]), name='w6')
    b6 = tf.Variable(zeroInitializer([n_classes]), name = 'b6')
    
    #Data and targets    
    x = tf.placeholder(tf.float32, shape=(None, 30,30,3), name='x')
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name='y')
    keep1 = tf.placeholder_with_default(1.0, shape=())
    keep2 = tf.placeholder_with_default(1.0, shape=())
    learning_rate_placeholder = tf.placeholder_with_default(0.001, shape=())

    #Defining the Network
    input_layer = x
    conv2d_1 = conv2d(input_layer, w1, b1)
    conv2d_2 = conv2d(conv2d_1, w2, b2)
    max_pooling2d_1 = max_pooling2d(conv2d_2)
    dropout_1 = dropout(max_pooling2d_1, keep_prob=keep1)
    conv2d_3 = conv2d(dropout_1, w3, b3)
    conv2d_4 = conv2d(conv2d_3, w4, b4)
    max_pooling2d_2 = max_pooling2d(conv2d_4)
    dropout_2 = dropout(max_pooling2d_2, keep_prob=keep1)
    flatten_1 = flatten(dropout_2, 3, 3, 64)
    dense_1 = dense(flatten_1, w5, b5, activation='relu')
    dropout_3 = dropout(dense_1, keep_prob=keep2)
    out = dense(dropout_3, w6, b6, activation='none')
    softmax_layer = tf.nn.softmax(out, name='softmax_layer')
    
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
    correct_pred = tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    loss_summary = tf.summary.scalar(name='Loss', tensor=loss_op)
    accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)

    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder, name='adam')
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    
    init = tf.global_variables_initializer()
        

    #initialize the datasets
    X, trainTargets,Xvalid,validationTargets=loadSplitTrainValidation()
    print("Data loaded")
       
    #prepareTables
    num_examples=trainTargets.shape[0]
    number_of_batches = num_examples//batch_size
    number_of_training = number_of_batches * epochs_to_run    
    
    trainLoss = np.zeros((number_of_training, 2))
    trainAccuracy = np.zeros((number_of_training, 2))
    validationLoss = np.zeros((epochs_to_run, 2))
    validationAccuracy = np.zeros((epochs_to_run, 2))

    #put tensors in a collection to retrieve in later trainings
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('optimizer', optimizer)
    tf.add_to_collection('loss_summary', loss_summary)
    tf.add_to_collection('accuracy_summary', accuracy_summary)
    tf.add_to_collection('loss_op', loss_op)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('out', out)
    tf.add_to_collection('softmax_layer', softmax_layer)
    tf.add_to_collection('dropout_1', dropout_1)
    tf.add_to_collection('dropout_2', dropout_2)
    tf.add_to_collection('dropout_3', dropout_3)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('global_step', global_step)
    tf.add_to_collection('keep1', keep1)
    tf.add_to_collection('keep2', keep1)
    tf.add_to_collection('learning_rate', learning_rate_placeholder)
    
    
    saver=tf.train.Saver(max_to_keep=3)    
    
    with tf.Session() as sess:
        #writers for loss and accuracy
        log_dir=os.path.join(cur_path, "logs")
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries_'+run_name), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries_'+run_name), sess.graph)
        
        sess.run(init)
        print("Variables initialized")


        for epoch in range(epochs_to_run):
            #Shuffle before each epcoh
            X,trainTargets = shuffle(X, trainTargets)
            
            print("----------------------------------------")
            print("Epoch ", (epoch+1))
            for minibatch_index in range(number_of_batches):
                print('\rMinibatch {}/{}'.format(minibatch_index, number_of_batches), end='\r')
                #select minibatch and run optimizer
                minibatch_x = X[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]
                minibatch_y = trainTargets[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]           
                feed_dict={x: minibatch_x, y: minibatch_y, keep1:keep1_prob, keep2:keep2_prob, learning_rate_placeholder:learning_rate}
                _, loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val  = sess.run([train_op, loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict=feed_dict)

                train_writer.add_summary(loss_summary_val, global_step_val)
                train_writer.add_summary(accuracy_summary_val, global_step_val)
                trainLoss[global_step_val-1]=(loss_val, global_step_val)
                trainAccuracy[global_step_val-1]=(accuracy_val, global_step_val)                
                
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            #validation loss and accuracy at the end of each epoch
            if(epoch==epochs_to_run-1 or not(onlyFinalValidationStats)):
                print("Start of validation")                
                feed_dict={x: Xvalid, y: validationTargets}
                loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val, = sess.run([loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict=feed_dict)
                print("Validation phase finished")

                eval_writer.add_summary(loss_summary_val, global_step_val)
                eval_writer.add_summary(accuracy_summary_val, global_step_val)
                validationLoss[epoch]=(loss_val, global_step_val)
                validationAccuracy[epoch]=(accuracy_val, global_step_val)
                
                print("Epoch " + str(epoch+1) + ", Validation Loss=" +\
                              "{:.8f}".format(loss_val) + ", Validation Accuracy= " + \
                              "{:.8f}".format(accuracy_val) + "\n")
                
            else:
                print("Epoch " + str(epoch+1) + " (no data, only final losses and accuracies will be saved)")
            
            global_step_val = sess.run(global_step)
            directory='./savedModels/'+run_name+"/state_at_step"
            save_path=saver.save(sess, directory, global_step=global_step_val)
            print("Model saved in path: %s" % save_path)
                
        print("Optimization Finished!")

    return trainLoss, trainAccuracy, validationLoss, validationAccuracy

def restoreGraphAndTrain(run_name, epochs_to_run, trainLoss, trainAccuracy, validationLoss, validationAccuracy, latestMetaFile, batch_size=64, onlyFinalValidationStats=False, keep1_prob=0.75, keep2_prob=0.5, learning_rate=0.001):
    """Trains an already trained model for traffic sign recognition
    
    Parameters
    ----------
    run_name : str
        name of the model
    epochs_to_run : int
        epochs to run in the training
    trainLoss : numpy array
        loss on the train batches throughout training, recorded after each minibatch
    trainAccuracy  : numpy array
        accuracy on the train batches throughout training, recorded after each minibatch
    validationLoss : numpy array
        loss on the validation dataset throughout training, recorded after each epoch
    validationAccuracy  : numpy array
        accuracy on the validation dataset throughout training, recorded after each epoch
    latestMetaFile : str
        ppath to the latest .meta file of this model
    batch_size : int, optional
        size of batches used for training (default is 64)    
    onlyFinalValidationStats : bool, optional
        takes validation loss and accuracy only at the end of training
        used to save time (default is False)      
    keep1_prob : float, optional
        keep probability of the first dropout layer (default is 0.75)    
    keep2_prob : float, optional
        keep probability of the second dropout layer (default is 0.5)    
    learning_rate : float, optional
        learning rate of the adam optimizer (default is 0.001)    
        
    Returns
    -------
    trainLoss : numpy array
        loss on the train batches throughout training, recorded after each minibatch
    trainAccuracy  : numpy array
        accuracy on the train batches throughout training, recorded after each minibatch
    validationLoss : numpy array
        loss on the validation dataset throughout training, recorded after each epoch
    validationAccuracy  : numpy array
        accuracy on the validation dataset throughout training, recorded after each epoch
    """  

    tf.set_random_seed(421)

    #initialize the datasets
    X, trainTargets,Xvalid,validationTargets=loadSplitTrainValidation()
    print("Data loaded")
    
    #prepareTables
    num_examples=trainTargets.shape[0]
    number_of_batches = num_examples//batch_size
    number_of_training = number_of_batches * epochs_to_run    
    
    trainLossNew = np.zeros((number_of_training, 2))
    trainAccuracyNew = np.zeros((number_of_training, 2))
    validationLossNew = np.zeros((epochs_to_run, 2))
    validationAccuracyNew = np.zeros((epochs_to_run, 2))

    # delete the current graph
    tf.reset_default_graph()
    # import the graph from the file
    tf.train.import_meta_graph('./savedModels/'+run_name+"/"+latestMetaFile)
    saver=tf.train.Saver(max_to_keep=3) 
    with tf.Session() as sess:
        #writers for loss and accuracy
        log_dir=os.path.join(cur_path, "logs")
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries_'+run_name), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries_'+run_name), sess.graph)
        
        #Get the latest checkpoint in the directory
        restore_from = tf.train.latest_checkpoint("./savedModels/"+run_name)
        #Reload the weights into the variables of the graph
        saver.restore(sess, restore_from)
        print("Variables initialized")
        
        #get access to the tensors
        train_op=tf.get_collection('train_op')[0]
        loss_summary=tf.get_collection('loss_summary')[0]
        accuracy_summary=tf.get_collection('accuracy_summary')[0]
        loss_op=tf.get_collection('loss_op')[0]
        accuracy=tf.get_collection('accuracy')[0]
#        softmax_layer=tf.get_collection('softmax_layer')[0]
        x= tf.get_collection('x')[0]
        y=tf.get_collection('y')[0]
        global_step = tf.get_collection('global_step')[0]
        keep1 = tf.get_collection('keep1')[0]
        keep2 = tf.get_collection('keep2')[0]
        learning_rate_placeholder = tf.get_collection('learning_rate')[0]
            
        global_step_val=sess.run(global_step)
        step_in_current_training_session = 0
        
        currentEpoch = global_step_val//number_of_batches
        print('Starting after ',global_step_val, 'steps (', currentEpoch, ' epochs)')
        
        for epoch in range(epochs_to_run):
            #Shuffle before each epcoh
            X,trainTargets = shuffle(X, trainTargets)
            
            print("----------------------------------------")
            print("Epoch ", (epoch+1))
            for minibatch_index in range(number_of_batches):
                print('\rMinibatch {}/{}'.format(minibatch_index, number_of_batches), end='\r')
                #select minibatch and run optimizer
                minibatch_x = X[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]
                minibatch_y = trainTargets[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]  
                feed_dict={x: minibatch_x, y: minibatch_y, keep1:keep1_prob, keep2:keep2_prob, learning_rate_placeholder:learning_rate}
                feed_dict={x: minibatch_x, y: minibatch_y, keep1:keep1_prob, keep2:keep2_prob, learning_rate_placeholder:learning_rate}
                _, loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val  = sess.run([train_op, loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict=feed_dict)

                train_writer.add_summary(loss_summary_val, global_step_val)
                train_writer.add_summary(accuracy_summary_val, global_step_val)
                trainLossNew[step_in_current_training_session-1]=(loss_val, global_step_val)
                trainAccuracyNew[step_in_current_training_session-1]=(accuracy_val, global_step_val)                
                step_in_current_training_session+=1
                
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            if(epoch==epochs_to_run-1 or not(onlyFinalValidationStats)):
                print("Start of validation")
                feed_dict={x: Xvalid, y: validationTargets}
                loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val, = sess.run([loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict=feed_dict)
                print("End validation")
                
                eval_writer.add_summary(loss_summary_val, global_step_val)
                eval_writer.add_summary(accuracy_summary_val, global_step_val)
                validationLossNew[epoch]=(loss_val, global_step_val)
                validationAccuracyNew[epoch]=(accuracy_val, global_step_val)
                
                print("Epoch " + str(epoch+1) + ", Validation Loss=" +\
                              "{:.8f}".format(loss_val) + ", Validation Accuracy= " + \
                              "{:.8f}".format(accuracy_val) + "\n")
                
            else:
                print("Epoch " + str(epoch+1) + " (no data, only final losses and accuracies will be saved)")
            global_step_val = sess.run(global_step)
            directory='./savedModels/'+run_name+"/state_at_step"
            save_path=saver.save(sess, directory, global_step=global_step_val)
            print("Model saved in path: %s" % save_path)
                
        print("Optimization Finished!")
    
    #Add logged data to previous ones
    trainLoss=np.concatenate((trainLoss, trainLossNew), axis=0)
    trainAccuracy=np.concatenate((trainAccuracy, trainAccuracyNew), axis=0)
    validationLoss=np.concatenate((validationLoss, validationLossNew), axis=0)
    validationAccuracy=np.concatenate((validationAccuracy, validationAccuracyNew), axis=0)
    
    return trainLoss, trainAccuracy, validationLoss, validationAccuracy        
  
    
def mainStartLearning(run_name, epochs_to_run=15, learning_rate=0.001):
    """Starts training a new model
     
    Parameters
    ----------
    run_name : str
        name of the model
    epochs_to_run : int, optional
        epochs to run in the training (default is 15)
    learnin_rate : float, optional
        learning rate of the Adam optimizer (default is 0.001)        
    """ 
    
    trainLoss, trainAccuracy, validationLoss, validationAccuracy = cnnTF(run_name, epochs_to_run=epochs_to_run, learning_rate=learning_rate)
    if not os.path.exists("pickleLogs"):
        os.makedirs("pickleLogs")
    picleName="pickleLogs/"+run_name+".pkl"
    with open(picleName, 'wb') as f: 
        pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy], f)
    return

def continueLearning(run_name, latestMetaFile, epochs_to_run=5, learning_rate=0.001):
    """Training a model that has already been trained
     
    Parameters
    ----------
    run_name : str
        name of the model
    latest_meta_file : str
        path to the latest .meta file saved for this model
    epochs_to_run : int, optional
        epochs to run in the training (default is 5)
    learnin_rate : float, optional
        learning rate of the Adam optimizer (default is 0.001)           
    """ 

    if not os.path.exists("pickleLogs"):
        os.makedirs("pickleLogs")
    pickleName = "pickleLogs/"+run_name+".pkl"
    with open(pickleName, 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)      
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = restoreGraphAndTrain(run_name, epochs_to_run, trainLoss, trainAccuracy, validationLoss, validationAccuracy, latestMetaFile, learning_rate=learning_rate)
        with open(pickleName, 'wb') as f: 
            pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy], f)
    return

def plotLogsAccuracy(run_name, startIndex, endIndex):
    """Plots accuracies logged throughout the training
    
    Plots both train accuracies which are logged after each epoch and
    validation accuracies logged after each epoch
     
    Parameters
    ----------
    run_name : str
        name of the model
    startIndex : int
        step at which to start plotting, each index corresponds to a trained minibatch
    endIndex : int
        step at which to end plotting, each index corresponds to a trained minibatch     
    """ 

    pickleName = "pickleLogs/"+run_name+".pkl"
    with open(pickleName, 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)
        
    startIndex = startIndex
    endIndex = endIndex
    
    plt.title("Accuracies")
    plt.plot(trainAccuracy[startIndex:endIndex,1],trainAccuracy[startIndex:endIndex,0], '-b', label='TrainData')
    plt.plot(validationAccuracy[startIndex:endIndex,1],validationAccuracy[startIndex:endIndex,0], '-r', label='ValidationData')
    plt.legend(loc='best')
    plt.xlabel('Training Steps')
    plt.show()
    return

def plotLogsLosses(run_name, startIndex, endIndex):
    """Plots losses logged throughout the training
    
    Plots both train losses which are logged after each epoch and
    validation losses logged after each epoch
     
    Parameters
    ----------
    run_name : str
        name of the model
    startIndex : int
        step at which to start plotting, each index corresponds to a trained minibatch
    endIndex : int
        step at which to end plotting, each index corresponds to a trained minibatch     
    """ 

    pickleName = "pickleLogs/"+run_name+".pkl"
    with open(pickleName, 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)
        
    startIndex = startIndex
    endIndex = endIndex
    plt.title("Losses")
    plt.plot(trainLoss[startIndex:endIndex, 1],trainLoss[startIndex:endIndex,0], '-b', label='TrainData')
    plt.plot(validationLoss[startIndex:endIndex,1],validationLoss[startIndex:endIndex,0], '-r', label='ValidationData')
    plt.legend(loc='best')
    plt.xlabel('Training Steps')
    plt.show()
    return

def predict(images, classes):
    """Predicts the classes of an array of images
     
    Parameters
    ----------
    images : numpy array
        array with images
    classes : numpy array
        array with the correct classes of each image, one-hot encoded 
    """ 

    images = np.reshape(images,(-1, 30,30,3))
    classes = np.reshape(classes,(-1, 43))
    # delete the current graph
    tf.reset_default_graph()
    # import the graph from the file
    
    tf.train.import_meta_graph('./savedModels/persistentMeta/trafficSignClassifier.meta')
    saver=tf.train.Saver(max_to_keep=3) 
    with tf.Session() as sess:        
        #Get the latest checkpoint in the directory
        restore_from = tf.train.latest_checkpoint("./savedModels")
        #Reload the weights into the variables of the graph
        saver.restore(sess, restore_from)
        print("Variables initialized")
        
        loss_op=tf.get_collection('loss_op')[0]
        accuracy=tf.get_collection('accuracy')[0]
        softmax_layer=tf.get_collection('softmax_layer')[0]
        x= tf.get_collection('x')[0]
        y=tf.get_collection('y')[0]

        feed_dict={x: images, y: classes}
        loss_val, accuracy_val, softmax_layer_val  = sess.run([loss_op, accuracy, softmax_layer], feed_dict=feed_dict)
        predicted_classes = np.argmax(softmax_layer_val, 1)
        correct_classes = np.argmax(classes,1)

        index_row = np.arange(len(softmax_layer_val))
        indices = list(zip(index_row, predicted_classes, correct_classes))
        for i in indices:
#            if i[1]!=i[2]:
            print("Image ", i[0], "is of class ", i[1], " with probability", softmax_layer_val[i[0:2]], " - correct class: ", i[2])
#            img = Image.fromarray(images[i[0]], 'RGB')
#            img.show()
        return


def cnnKeras():
    """NN model defined in Keras
    """ 
   
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

def askYesNo(question):
    """Asks a yes/no question to the user
     
    Parameters
    ----------
    question : str
        question to ask to the user

    Returns
    -------
    bool
        True if user said yes else False
    """ 

    answer=""
    while not(answer=="y" or answer=="n"):
        answer=input(question+" [y/n]:") 
    return True if answer=="y" else False

def askForFloat(question, minValue, maxValue):
    """Asks the user for a float
     
    Parameters
    ----------
    question : str
        question to ask to the user
    minValue : float
        minimum accepted value
    maxValue : float
        maximum accepted value

    Returns
    -------
    float
        float given by the user
    """ 

    while True:
        try:
            value=float(input(question))
        except ValueError:
            print("This is not a float")
        if(value<minValue or value>maxValue):
            print("The value has to be between {} and {}".format(minValue, maxValue))
        else:
            return value

def askForInteger(question, minValue, maxValue):
    """Asks the user for an int
     
    Parameters
    ----------
    question : str
        question to ask to the user
    minValue : int
        minimum accepted value
    maxValue : int
        maximum accepted value

    Returns
    -------
    int
        int given by the user
    """ 

    while True:
        try:
            value=int(input(question))
        except ValueError:
            print("This is not a number")
        if(value<minValue or value>maxValue):
            print("The value has to be between {} and {}".format(minValue, maxValue))
        else:
            return value    

def main():
    """Main function of the training of the model

    Asks the user which model to train.
    If the model has already been trained, continue training the model
    Otherwise creates the model
    """ 

    run_name="trafficSignClassifier"

    print("The default name for the trained model is {}.".format(run_name))
    useDefaultName=askYesNo("Do you want to use the default name?")
    
    if not(useDefaultName):
        confirmed=False
        while not(confirmed):
            run_name=input("Input name of model:")
            val, file_name=findLatestMetaFile(run_name)    
            if val==-1:
                print("{} has not been created".format(run_name))
                confirmed=askYesNo("Do you want to train this new model?")
            else:
                print("{} has already been trained and its current train step is {}".format(run_name, val))
                confirmed=askYesNo("Do you want to continue training this model?")
    
    epochs_to_run=askForInteger("Number of epochs to run: ", 1, 100)
    learning_rate=askForFloat("Learning rate: ", 1e-20, 1-1e-20)
        
    val, file_name=findLatestMetaFile(run_name)
    if val!=-1:
        print("Continue learning after step {}".format(val))
        continueLearning(run_name, file_name, epochs_to_run=epochs_to_run, learning_rate=learning_rate)
    else:
        print("Start learning from scratch")
        mainStartLearning(run_name, epochs_to_run=epochs_to_run, learning_rate=learning_rate)
    
if __name__=="__main__":
    main()

