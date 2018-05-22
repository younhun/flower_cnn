import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf

img_width, img_height = 224, 224
model_path = './models2/model.h5'
model_weights_path = './models2/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

model.summary()
# keras 와 tensorflow graph충돌
#############################
graph = tf.get_default_graph()
#############################

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)

  
  with graph.as_default():
    array = model.predict(x)
  
  print(array)
  # one-hot encoding)
  result = array[0]
  
  print(result)
  # one-hot encoding 결과 값 추출
  # 각 결과 값 label에 matching
  answer = np.argmax(result)
  return answer
