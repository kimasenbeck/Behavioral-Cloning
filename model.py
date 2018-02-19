
import os
import csv
import cv2
import numpy as np
import sklearn
#18726
lines = []
with open('examples\driving_log.csv') as csvfile:
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
                correction = 0.2
                for i in range(1):
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    #Preprocess image
                    image = cv2.GaussianBlur(image, (3,3), 0)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    images.append(image)
                    
                measurement = float(line[3])
                angles.append(measurement)
                #angles.append(measurement + correction) # adjust for left camera
                #angles.append(measurement - correction) # adjust for right camera

            
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

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch= 2*len(train_samples),
                    validation_data=validation_generator, 
                    nb_val_samples=6*len(validation_samples), 
                    nb_epoch=2, verbose=1)

model.save('model.h5') 
exit()