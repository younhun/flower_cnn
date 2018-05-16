import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 128, 128
model_path = './models/model.h5'
# model_weights_path = './models/weights.h5'
model = load_model(model_path)
# model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  print(x.shape)

  # one-hot encoding
  array = model.predict(x)
  result = array[0]
  
  # one-hot encoding 결과 값 추출
  # 각 결과 값 label에 matching
  answer = np.argmax(result)
  if answer == 0:
    print("Label: 개나리")
  elif answer == 1:
    print("Label: 동백")
  elif answer == 2:
    print("Label: 목화")
  elif answer == 3:
    print("Label: 백합")
  elif answer == 4:
    print("Label: 안투리움")
  elif answer == 5:
    print("Label: 장미")
  return answer


n0_t = 0
n0_f = 0
n1_t = 0
n1_f = 0
n2_t = 0
n2_f = 0
n3_t = 0
n3_f = 0
n4_t = 0
n4_f = 0
n5_t = 0
n5_f = 0


for i, ret in enumerate(os.walk('./test-data/개나리')):
  for i, filename in enumerate(ret[2]):
    # ret[2] = folder에 들어있는 images들을 가리킨다.
    
    # DS_Store 처리
    if filename.startswith("."):
      continue
    
    # ret[0] = image 경로
    # ret[0] + / + filename = 경로에 들어있는 image
    # 경로의 image를 predict 함수에 넣는다.
    result = predict(ret[0] + '/' + filename)
    
    # result에 따라 true, false 
    if result == 0:
      n0_t += 1
    else:
      n0_f += 1

for i, ret in enumerate(os.walk('./test-data/동백')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      n1_t += 1
    else:
      n1_f += 1

for i, ret in enumerate(os.walk('./test-data/목화')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      n2_t += 1
    else:
      n2_f += 1

for i, ret in enumerate(os.walk('./test-data/백합')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      n3_t += 1
    else:
      n3_f += 1

for i, ret in enumerate(os.walk('./test-data/안투리움')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      n4_t += 1
    else:
      n4_f += 1

for i, ret in enumerate(os.walk('./test-data/장미')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      n5_t += 1
    else:
      n5_f += 1

# """
# Check metrics
# """
print("True 개나리: ", n0_t)
print("False 개나리: ", n0_f)
print("True 동백: ", n1_t)
print("False 동백: ", n1_f)
print("True 목화: ", n2_t)
print("False 목화: ", n2_f)
print("True 백합: ", n3_t)
print("False 백합: ", n3_f)
print("True 안투리움: ", n4_t)
print("False 안투리움: ", n4_f)
print("True 장미: ", n5_t)
print("False 장미: ", n5_f)