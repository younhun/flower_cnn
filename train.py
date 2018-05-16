import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import callbacks
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras import applications
from keras import regularizers


DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == '--development' or argvs[1] == '-d'):
	DEV = True

if DEV:
	epochs = 2
else:
	epochs = 20


train_data_path = './data/train'
validation_data_path = './data/validation'

train_data_len = 0
validation_data_len = 0

for _, _, files in os.walk(train_data_path):
	train_data_len += len(files)

for _, _, files in os.walk(validation_data_path):
	validation_data_len += len(files)

img_height = 128
img_width = 128
input_shape = (img_height, img_width, 3)
chanDim = -1

classes_num = 6
batch_size = 32
# samples_per_epoch = train_data_len // batch_size
samples_per_epoch = 100
# validation_steps = validation_data_len // batch_size
validation_steps = 30


lr = 0.0004

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
model.add(Dense(classes_num))
model.add(Activation('softmax'))

opt = Adam(lr=1e-3, decay=1e-3 / epochs)

model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Data pre_processing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. /255)


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

# training model with generator
# training
model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=cbks,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

# Save model
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')


