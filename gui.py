# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:05:40 2020

@author: Alexander
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import Button, Label, BOTTOM
from PIL import ImageTk, Image
from random import randint
from settings import cur_path, classes_dict
import tensorflow as tf
import numpy as np
import os
import pandas as pd

def main():
    #load the trained model to classify sign
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
        
        test_file = pd.read_csv('Test.csv')
        testTarget=test_file["ClassId"].values
        testTarget = np.array(testTarget)
        
        #initialise GUI
        top=tk.Tk()
        top.geometry('800x600')
        top.title('Traffic sign classification')
        top.configure(background='#e2e8e9')
        label=Label(top,background='#e2e8e9', font=('arial',15,'bold'))
        correct_label = Label(top,background='#e2e8e9', font=('arial',15,'bold'))
        sign_image = Label(top)
        
        def classify(file_path, correct_pred=None):
            global label_packed
            image = Image.open(file_path)
            image = image.resize((30,30))
            image = np.array(image)     
            image = np.reshape(image, (-1, 30, 30, 3))
            feed_dict={x: image}
            softmax_layer_val  = sess.run(softmax_layer, feed_dict=feed_dict)
            pred = np.argmax(softmax_layer_val, 1)[0] 
            pred_prob=softmax_layer_val[0,pred]
            sign = classes_dict[pred+1]
            label.configure(foreground='#011638', text="Predicted class: "+sign+" with probability "+"{:.4f}".format(pred_prob)) 
            if correct_pred!=None:  
                foreground_color='#267f1a' #green
                if pred!=correct_pred:
                    foreground_color = '#911608' #red
                correct_sign = classes_dict[correct_pred+1]                
                correct_label.configure(foreground=foreground_color, text="Correct class: "+correct_sign)
                label.configure(foreground=foreground_color)
                            
        def show_classify_button(file_path, correct_pred=None):
            classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path, correct_pred),padx=10,pady=5)
            classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
            classify_b.place(relx=0.79,rely=0.46)
            
        def upload_image():
            try:
                file_path=filedialog.askopenfilename()
                uploaded=Image.open(file_path)
                uploaded.thumbnail(((top.winfo_width()),(top.winfo_height())))
                im=ImageTk.PhotoImage(uploaded)
                sign_image.configure(image=im)
                sign_image.image=im
                label.configure(text='')
                correct_label.configure(text='')
                show_classify_button(file_path)
            except:
                pass
            
        def upload_Test_image():
            try:
                file_path=filedialog.askopenfilename()
                uploaded=Image.open(file_path)
                uploaded.thumbnail(((top.winfo_width()),(top.winfo_height())))
                im=ImageTk.PhotoImage(uploaded)
                sign_image.configure(image=im)
                sign_image.image=im
                label.configure(text='')
                correct_label.configure(text='')
                try:
                    base=os.path.basename(file_path)
                    integer_as_string=base[:5]
                    integer = int(integer_as_string)
                    if integer >=0 and integer <= 12629:
                        correct_pred=testTarget[integer]
                        show_classify_button(file_path, correct_pred=correct_pred)
                    else:
                        show_classify_button(file_path)
                except:
                    print("cant extract classs")
                    show_classify_button(file_path)
                    return
                
    
            except:
                pass        
        
        def upload_random_image():
            max_image=12629
            try:
                value = randint(0, max_image)
                name = '{0:05d}'.format(value) + ".png"
                file_path=os.path.join(cur_path, "Test")
                file_path=os.path.join(file_path, name)
                uploaded=Image.open(file_path)
                uploaded.thumbnail(((top.winfo_width()),(top.winfo_height())))
                im=ImageTk.PhotoImage(uploaded)
                sign_image.configure(image=im)
                sign_image.image=im
                label.configure(text='')
                correct_label.configure(text='')
                
                correct_pred=testTarget[value]
                show_classify_button(file_path, correct_pred=correct_pred)
            except:
                print("error while loading random image")
                pass
        
        upload=Button(top,text="Upload an image not from Test dataset",command=upload_image,padx=10,pady=5)
        upload.configure(background='#795d66', foreground='white',font=('arial',10,'bold'))
        upload.pack(side=BOTTOM,pady=5)
        
        uploadTest=Button(top,text="Upload an image from Test dataset",command=upload_Test_image,padx=10,pady=5)
        uploadTest.configure(background='#5c7c5a', foreground='white',font=('arial',10,'bold'))
        uploadTest.pack(side=BOTTOM, pady=5)
        
        uploadRand=Button(top,text="Upload a random image from Test dataset and check",command=upload_random_image,padx=10,pady=5)
        uploadRand.configure(background='#5c7c5a', foreground='white',font=('arial',10,'bold'))
        uploadRand.pack(side=BOTTOM, pady=5)
        
        
        sign_image.pack(side=BOTTOM,expand=True)
        label.pack(side=BOTTOM,expand=True)
        correct_label.pack(side=BOTTOM,expand=True)
        heading = Label(top, text="Use trained Model for traffic signs",pady=20, font=('arial',20,'bold'))
        heading.configure(background='#e2e8e9',foreground='#364156')
        heading.pack()
        top.mainloop()
    return

if __name__ == "__main__":
    main()



