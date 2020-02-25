# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:02:33 2020
@author: AlexanderApostolov
"""
import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from settings import classes, cur_path
from dataLoadingAndManipulation import shuffle, loadSplitTrainValidation, loadTestData
from layersNN import conv2d, max_pooling2d, dropout, flatten, dense
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
To view logged values of accuracy and loss
>>>tensorboard --logdir="./logs" --port 6006
To see results go to http://localhost:6006/#scalars
"""

def cnnTF(batch_size=64, epochs_to_run=15, onlyFinal=False, n_classes=classes, keep1_prob=0.75, keep2_prob=0.5):
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
    learning_rate = tf.placeholder_with_default(0.001, shape=())

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

    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam')
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
    tf.add_to_collection('learning_rate', learning_rate)
    
    
    saver=tf.train.Saver(max_to_keep=3)    
    
    with tf.Session() as sess:
        log_dir=os.path.join(cur_path, "logs")
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries'), sess.graph)
        
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
                _, loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val  = sess.run([train_op, loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict={x: minibatch_x, y: minibatch_y, keep1:keep1_prob, keep2:keep2_prob})

                train_writer.add_summary(loss_summary_val, global_step_val)
                train_writer.add_summary(accuracy_summary_val, global_step_val)
                trainLoss[global_step_val-1]=(loss_val, global_step_val)
                trainAccuracy[global_step_val-1]=(accuracy_val, global_step_val)                
                
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            if(epoch==epochs_to_run-1 or not(onlyFinal)):
                print("Start of validation")                
                loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val, = sess.run([loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict={x: Xvalid, y: validationTargets})
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
            save_path=saver.save(sess, './savedModels/trafficSignClassifier', global_step=global_step_val)
            print("Model saved in path: %s" % save_path)
                
        print("Optimization Finished!")

    return trainLoss, trainAccuracy, validationLoss, validationAccuracy

def restoreGraphAndTrain(epochs_to_run, trainLoss, trainAccuracy, validationLoss, validationAccuracy, batch_size=64, onlyFinal=False, keep1_prob=0.75, keep2_prob=0.5):
    tf.set_random_seed(421)

#    initialize the datasets
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
    tf.train.import_meta_graph('./savedModels/persistentMeta/trafficSignClassifier.meta')
    saver=tf.train.Saver(max_to_keep=3) 
    with tf.Session() as sess:
        log_dir=os.path.join(cur_path, "logs")
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries'), sess.graph)
        
        #Get the latest checkpoint in the directory
        restore_from = tf.train.latest_checkpoint("./savedModels")
        #Reload the weights into the variables of the graph
        saver.restore(sess, restore_from)
        print("Variables initialized")
        
        train_op=tf.get_collection('train_op')[0]
        loss_summary=tf.get_collection('loss_summary')[0]
        accuracy_summary=tf.get_collection('accuracy_summary')[0]
        loss_op=tf.get_collection('loss_op')[0]
        accuracy=tf.get_collection('accuracy')[0]
        softmax_layer=tf.get_collection('softmax_layer')[0]
        x= tf.get_collection('x')[0]
        y=tf.get_collection('y')[0]
        global_step = tf.get_collection('global_step')[0]
        keep1 = tf.get_collection('keep1')[0]
        keep2 = tf.get_collection('keep2')[0]
        learning_rate = tf.get_collection('learning_rate')[0]
            
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
                feed_dict={x: minibatch_x, y: minibatch_y, keep1:keep1_prob, keep2:keep2_prob}
                _, loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val  = sess.run([train_op, loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict=feed_dict)

                train_writer.add_summary(loss_summary_val, global_step_val)
                train_writer.add_summary(accuracy_summary_val, global_step_val)
                trainLossNew[step_in_current_training_session-1]=(loss_val, global_step_val)
                trainAccuracyNew[step_in_current_training_session-1]=(accuracy_val, global_step_val)                
                step_in_current_training_session+=1
                
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            if(epoch==epochs_to_run-1 or not(onlyFinal)):
                print("Start of validation")
                loss_val, accuracy_val, loss_summary_val, accuracy_summary_val, global_step_val, = sess.run([loss_op, accuracy, loss_summary, accuracy_summary, global_step], feed_dict={x: Xvalid, y: validationTargets})
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
            save_path=saver.save(sess, './savedModels/trafficSignClassifier', global_step=global_step_val)
            print("Model saved in path: %s" % save_path)
                
        print("Optimization Finished!")
    
    #Add logged data to previous ones
    trainLoss=np.concatenate((trainLoss, trainLossNew), axis=0)
    trainAccuracy=np.concatenate((trainAccuracy, trainAccuracyNew), axis=0)
    validationLoss=np.concatenate((validationLoss, validationLossNew), axis=0)
    validationAccuracy=np.concatenate((validationAccuracy, validationAccuracyNew), axis=0)
    
    return trainLoss, trainAccuracy, validationLoss, validationAccuracy        
  
    
def mainStartLearning(epochs_to_run=15):
    trainLoss, trainAccuracy, validationLoss, validationAccuracy = cnnTF(epochs_to_run=epochs_to_run)
    with open('optTrafficSigns.pkl', 'wb') as f: 
        pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy], f)
    return

def continueLearning(epochs_to_run=5):
    with open('optTrafficSigns.pkl', 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)      
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = restoreGraphAndTrain(epochs_to_run, trainLoss, trainAccuracy, validationLoss, validationAccuracy)
        with open('optTrafficSigns.pkl', 'wb') as f: 
            pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy], f)
    return

def plotLogsAccuracy():
    with open('optTrafficSigns.pkl', 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)
        
    startIndex = 0
    endIndex = 7000
    
    plt.title("Accuracies")
    plt.plot(trainAccuracy[startIndex:endIndex,1],trainAccuracy[startIndex:endIndex,0], '-b', label='TrainData')
    plt.plot(validationAccuracy[startIndex:endIndex,1],validationAccuracy[startIndex:endIndex,0], '-r', label='ValidationData')
    plt.legend(loc='best')
    plt.xlabel('Training Steps')
    plt.show()
    return

def plotLogsLosses():
    with open('optTrafficSigns.pkl', 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy = pickle.load(f)
        
    startIndex = 0
    endIndex = 1000
    plt.title("Losses")
    plt.plot(trainLoss[startIndex:endIndex, 1],trainLoss[startIndex:endIndex,0], '-b', label='TrainData')
    plt.plot(validationLoss[startIndex:endIndex,1],validationLoss[startIndex:endIndex,0], '-r', label='ValidationData')
    plt.legend(loc='best')
    plt.xlabel('Training Steps')
    plt.show()
    return

def predict(images, classes):
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

        feed_dict={x: images[], y: classes[]}
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