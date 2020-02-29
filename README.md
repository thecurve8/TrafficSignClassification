# TrafficSignClassification
Neural network classifier implemented in Tensorflow. Classifies images of traffic signs into 43 classes. 

The function 
```python
cnnKeras()
```
defines the same network in keras.

The dataset used is "The German Traffic Sign Benchmark"
It is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.

You can download it [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

Models can be created and trained and then saved.
For later training you can restore the last state of the model and continue training with the function.

## Installation
The data set has to be in the folder with the project.

The environment needs to have tensorflow, numpy, pickle, matplotlib, PIL, pandas and tkinter

## The neural network
The network is as follows:
- 1 **input layer**: RGB images of size 30x30
- 2 **Convolutional layers** with 32 filters and kernel size (5,5) and activation ReLu
- 1 **MaxPool layer** with pool_size (2,2)
- 1 **Dropout layer** with keep_prob 0.75 (keep_prob is 1.0 when not used for training)
- 2 **Convolutional layers** with 64 filters and kernel size (3,3) and activation ReLu
- 1 **MaxPool layer** with pool_size (2,2)
- 1 **Dropout layer** with keep_prob 0.75 (keep_prob is 1.0 when not used for training)
- 1 **Flatten layer**
- 1 **Dense layer** with 256 outputs and activation ReLu
- 1 **Dropout layer** with keep_prob 0.5 (keep_prob is 1.0 when not used for training)
- 1 **Dense layer** with 43 outputs (all the classes) and activation softmax

### Initialization
The weights for the convolutional and dense layers are initialized with the *xavier uniform* distribution.
The biases for the convolutional and dense layers are initialized with *zeros*

### Loss
The loss is defined as the softmax cross entropy.

### Optimizer
the Adam optimizer is used with the following parameters:
- learning_rate : can be changed, default is 0.001
- beta1 : 0.9
- beta2 : 0.999
- epsilon : 1e-07

## Training
Training is done in batches of size 64 with the adam optimizer.

## Accuracy and loss
Accuracy and loss values are taken throughout the whole training process (even if different training sessions) and are logged.
These are saved by writers and also in numpy arrays in pickle files.

*Training* accuracy and loss are taken after each minbatch.
*Validation* accuracy and loss are taken at the end of each epoch.

There are two ways to view the logs:

### Tensorboard
1. run
```bash
tensorboard --logdir="./logs" --port 6006 
```
2. To see results go to http://localhost:6006/#scalars

### Numpy arrays in pickle file
You can view losses by running:
```python
plotLogsLoss(run_name, startIndex, endIndex)
```

You can view accuracies by running:
```python
plotLogsAccuracy(run_name, startIndex, endIndex)
```

These functions are in the classifierTrafficSigns.py script

## GUI
There is a GUI that can be used to see how the model predicts images of road signs.
You can either predict images from the Test dataset or load images from anywhere else.

![gui screenshot](https://user-images.githubusercontent.com/18367214/75614048-b0bfac80-5b34-11ea-8b66-afd92457773b.PNG)


## Usage
 To train a new model:
```bash
python classifierTrafficSigns.py
The default name for the trained model is trafficSignClassifier.
Do you want to use the default name? [y/n]:y
Input name of model:Test
Meta file not found (directory not found)
Test has not been created
Do you want to train this new model? [y/n]:y
Number of epochs to run: 15
Learning rate: 0.001 
```

To train a model that has already been trained:
```bash
python classifierTrafficSigns.py
The default name for the trained model is trafficSignClassifier.
Do you want to use the default name? [y/n]:n
Input name of model:Test
Biggest step found is  490
Meta file is state_at_step-490.meta
Test has already been trained and its current train step is 490
Do you want to continue training this model? [y/n]:y
Number of epochs to run: 3
Learning rate: 0.0001
```

## Example resuts
After training for 15 epochs with a learning rate of 0.001, the results are the following:

Accuracy : 0.9798

Loss : 0.07972

Here are the plots from Tensorboard, the blue line is the training data, the orange one is the validation data:

![accuracy plot](https://user-images.githubusercontent.com/18367214/75614004-27a87580-5b34-11ea-93b0-7809a101e075.PNG)
![loss plot](https://user-images.githubusercontent.com/18367214/75614015-4575da80-5b34-11ea-86da-a76312404582.PNG)

