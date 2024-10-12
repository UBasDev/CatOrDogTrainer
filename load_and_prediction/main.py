# Importing all necessary libraries

import random
from keras.api.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D
from keras.api.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

model = load_model('../model_saved.keras')
 
image_number = str(random.randint(1, 2500))
 
image = load_img('../test/dogs/' + image_number + '.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
predicted_class = 1 if label[0][0] >= 0.5 else 0
print("PREDICT - (0 Cats , 1 Dogs): ", label[0][0])
print("IMAGENUMBER: ", image_number)