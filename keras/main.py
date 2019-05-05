from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

width = 224
height = 224
vgg16_no_dense = VGG16(include_top=False, weights='imagenet', input_shape=(width, height, 3))

model = Flatten(name="flatten")(vgg16_no_dense.output)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)

model = Dense(1,activation='sigmoid')(model)
model_vgg = Model(vgg16_no_dense.input, model, name='vgg16')

model_vgg.summary()
