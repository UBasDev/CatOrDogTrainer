# Importing all necessary libraries
import os
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D
from keras.api.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras._tf_keras.keras import backend as K2
from keras.api.regularizers import l2
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
import numpy as np

class CyclicLR(Callback):
    def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=2000, mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.iteration = 0

    def on_train_begin(self, logs=None):
        # Set the initial learning rate
        self.model.optimizer.learning_rate.assign(self.base_lr)
        self.iteration = 0

    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            new_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))

        # Assign the new learning rate
        self.model.optimizer.learning_rate.assign(new_lr)
        print(f'Batch {self.iteration}: Adjusted learning rate to {new_lr:.6f}')

img_width, img_height = 224, 224

train_data_dir = 'train'
validation_data_dir = 'test'
nb_train_samples = 10000
nb_validation_samples = 2500
epochs = 30
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64,  kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

weights_path = 'model_saved.weights.keras'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights from:", weights_path)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Create model checkpoint callback to save the best model
checkpoint = ModelCheckpoint(filepath='model_saved.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
clr = CyclicLR(base_lr=1e-4, max_lr=1e-2, step_size=2000)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,   # New augmentations
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
    )
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint, early_stopping, clr, tensorboard]
    )

model.save_weights('model_saved.weights.h5')

# Print the class indices
print("Class indices:", train_generator.class_indices)
print("Class indices:", test_datagen.class_indices)