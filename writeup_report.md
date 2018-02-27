# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### In this writeup, I will consider each of the [rubric points](https://review.udacity.com/#!/rubrics/432/view) provided by Udacity.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [Video of successful lap around the track](https://youtu.be/mMmUEA8q554)
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the model works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a basic convolutional neural network.  

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting. 

To ensure that the model was not overfitting, I split my data into a training set and a validation set. The model was tested by running it through the simulator and ensuring that the vehicle makes it around the track without veering off the road.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I made use of the training data supplied by Udacity. More on this in the seciton below.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with a known, successful model. In my case, I began with the NVIDIA self driving car convolutional neural network. I thought this model might be appropriate because it's designed to run in scenarios exactly like this project. 

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle fell off the track, but everything else was flawless. Since I was randomizing data and also employing dropout, I suspected that re-training the model with the same data and architecture might resolve the issue. In fact, after retraining, my problem went away and the car drove successfully around the track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network that resembles the [NVIDIA self driving car convolutional neural network architecture](NVIDIA convolutional neural network architecture). This consists of several 2D convolutional layers, a flattening layer, and then a series of density layers which ultimately leave us with one result, the predicted steering angle. I augmented this model slightly by adding a dropout layer, and by playing around with activation functions. 


#### 3. Creation of the Training Set & Training Process

Creating the training set was the most challenging part of the project. In reality, the data collection in itself was not so complicated. To capture good driving behavior, I first recorded a lap on the track using center lane driving. I then did a few extra tracks to reinforce the turns, and then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to veer back to the center of the road if it started to go off track. 

In the end, though, I used the sample dataset provided by Udacity. I ended up spending too much time trying to generate the perfect dataset, and decided that I would be better off using Udacity's data as a known viable starting point, and then modifying my model based on that baseline. The Udacity dataset provides 8036 data points. I  preprocessed this data by cropping each of the images to exclude the area above the road. To augment the data sat, I flipped each of the images across the vertical axis. Doing so doubled the size of my dataset, and provided data for both clockwise and counterclockwise turns. Finally, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3, as evidenced by the resulting loss. Since I used an Adam optimizer, it wasn't necessary for me to manually tune the learning rate. 
