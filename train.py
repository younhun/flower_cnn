import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

from model import CNN2D


DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 200

train_data_path = './data/train'
validation_data_path = './data/validation'

train_data_len = 0
validation_data_len = 0

for _, _, files in os.walk(train_data_path):
    train_data_len = train_data_len + len(files)

for _, _, files in os.walk(validation_data_path):
    validation_data_len = validation_data_len + len(files)


"""
Parameters
"""
img_width, img_height = 80, 80
image_shape = (img_width, img_height, 3)
batch_size = 32

samples_per_epoch = train_data_len // batch_size
validation_steps = validation_data_len // batch_size

classes_num = 8
lr = 0.0002

model = CNN2D(classes_num, image_shape)

# Early stop
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=30, \
                          verbose=1, mode='auto')

# optimization details
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip = True)

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

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb, earlystop]


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
model.save_weights('./models/weights.h5')




