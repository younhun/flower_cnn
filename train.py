import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks

DEV = False
argvs = sys.argv
argc = len(argvs)


if drgc > 1 and (argvs[1] == '--development' or argvs[1] == '-d'):
	DEV = True

if DEV:
	epochs = 2
else:
	epochs = 20


train_data_path = './data/train'
validation_data_path = './data/validation'

img_width, img_height = 224, 224
classes_num = 13
lr = 0.0004

batch_size = 32
samples_per_epoch = 1000
validation_steps = 300

# VGG16
model = Sequential()

# Block1
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', input_shape=(img_width, img_height, 3), name='block1_conv1'))
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
# Block2
model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
# Block3
model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
# Block4
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
# Block5
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu',name='fc1'))
model.add(Dropout(0.5,name='dropout1'))
model.add(Dense(4096, activation='relu',name='fc2'))
model.add(Dropout(0.5,name='dropout2'))
model.add(Dense(classes_num, activation='softmax', name='predictions'))



model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

# Data 회전등 pre_processing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# Tensorboard log
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')



