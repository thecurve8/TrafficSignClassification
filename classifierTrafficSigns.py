# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:02:33 2020
@author: AlexanderApostolov
"""
import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from settings import classes, cur_path
from dataLoadingAndManipulation.py import shuffle, loadSplitTrainValidation
from layersNN import conv2d, max_pooling2d, dropout, flatten, dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Run main
>>>tensorboard --logdir="./logs" --port 6006
To see results go to http://localhost:6006/#scalars
"""


def cnnTF(keep1=0.75, keep2=0.5, learning_rate=0.001, batch_size=64, epochs=15, onlyFinal=False, n_classes=classes):
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
    softmax_layer = tf.nn.softmax(out)
    
    
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
    correct_pred = tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    loss_summary = tf.summary.scalar(name='Loss', tensor=loss_op)
    accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)

    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    
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
        log_dir=os.path.join(cur_path, "logs")
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries'), sess.graph)
        
        sess.run(init)
        print("Variables initialized")
        num_examples=trainTargets.shape[0]
        number_of_batches = num_examples//batch_size

        for step in range(epochs):
            #Shuffle before each epcoh
            X,trainTargets = shuffle(X, trainTargets)
            
            print("----------------------------------------")
            print("Epoch ", (step+1))
            for minibatch_index in range(number_of_batches):
                print('\rMinibatch {}/{}'.format(minibatch_index, number_of_batches), end='\r')
                #select minibatch and run optimizer
                minibatch_x = X[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]
                minibatch_y = trainTargets[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]           
                _, loss_summary_val, accuracy_summary_val, global_step_val  = sess.run([train_op, loss_summary, accuracy_summary, global_step], feed_dict={x: minibatch_x, y: minibatch_y})
                train_writer.add_summary(loss_summary_val, global_step_val)
                train_writer.add_summary(accuracy_summary_val, global_step_val)
                
                
            print('\rMinibatch {}/{}'.format(number_of_batches, number_of_batches), end='\r')
            print('\n')
            
            if(step==epochs-1 or not(onlyFinal)):
#                lossTrain, accTrain = sess.run([loss_op, accuracy], feed_dict={x: flat_x_shuffled, y: trainingLabels_shuffled})
                Xvalid, validationTargets = shuffle(Xvalid, validationTargets)
                XvalidSubset, validationTargetsSubstet = Xvalid[:640], validationTargets[:640]
                lossValid, accValid, loss_summary_val, accuracy_summary_val, sm_layer, = sess.run([loss_op, accuracy, loss_summary, accuracy_summary, softmax_layer], feed_dict={x: XvalidSubset, y: validationTargetsSubstet})
                eval_writer.add_summary(loss_summary_val, global_step_val)
                eval_writer.add_summary(accuracy_summary_val, global_step_val)
                
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