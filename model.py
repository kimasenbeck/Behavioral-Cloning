import os
import csv
import cv2
import numpy as np
import sklearn

#read in data
lines = []
with open('data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for line in batch_samples:
                #read in each image
                source_path = "C:\\Users\\kimasenbeck\\CarND-Behavioral-Cloning-P3\\data\\IMG\\" + line[i][4:]
                image = cv2.imread(source_path)

                #Preprocess image
                image = cv2.GaussianBlur(image, (3,3), 0)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                    
                #add measurements for each image
                measurement = float(line[3])
                angles.append(measurement)
 
            #augment dataset by flipping across axis
            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#construct network architecure
model = Sequential()
#begin by normalizing data
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
#crop images to exclude landscape
model.add(Cropping2D(cropping=((50,20), (0,0))))
#add several convolutions
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
#dropout to reduce overfitting
model.add(Dropout(0.3))
#reduce the current result to one final measurement, which is our predicted steering angle
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 2*len(train_samples),
                    validation_data=validation_generator, 
                    nb_val_samples=2*len(validation_samples), 
                    nb_epoch=3, verbose=1)

model.save('model.h5') 
exit()