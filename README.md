# TrafficSignClassification
Classifies 43 classes of traffic signs

The dataset used is "The German Traffic Sign Benchmark"
It is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.

You can download it at https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

This data set has to be in the folder with the python code.

Only Tensorflow is used to define the network. An equivalent network has been defined with keras.

The network is as follows:
- input: (-1, 30,30,3) - RGB images of size 30x30
- 2 Convolutional layers with 32 filters and kernel size (5,5) and activation ReLu
- 1 MaxPool layer with pool_size (2,2)
- 1 Dropout layer with keep_prob 0.75 (keep_prob is 1.0 when not used for training)
- 2 Convolutional layers with 64 filters and kernel size (3,3) and activation ReLu
- 1 MaxPool layer with pool_size (2,2)
- 1 Dropout layer with keep_prob 0.75 (keep_prob is 1.0 when not used for training)
- 1 Flatten layer
- 1 Dense layer with 256 outputs and activation ReLu
- 1 Dropout layer with keep_prob 0.5 (keep_prob is 1.0 when not used for training)
- 1 Dense layer with 43 outputs (all the classes) and activation softmax

The model can be created and trained and then saved.
For later training you can restore the last state of the model and continue training. 

There is a GUI that can be used to see how the model predicts images of road signs
