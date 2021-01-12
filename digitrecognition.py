import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

##split the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

##reshape the data to 4d array (samples*pixels*width*height) to fit keras api
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

##convert int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
##normalizing to range [0,1],(255 - RGB)
x_train /= 255.0
x_test /= 255.0

##build model
##he_uniform initializer - variance class - draws samples from a uniform distribution

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
##Flatten 2D to 1D before building fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

##compile and fit
##adam - best suited for CNN
##sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)

model.evaluate(x_test, y_test)

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")


