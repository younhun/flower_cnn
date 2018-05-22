import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

img_width = 224
img_height = 224

model_path = './models2/model.h5'
model_weights_path = './models2/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  
  # one-hot encoding
  array = model.predict(x)
  result = array[0]
  
  # one-hot encoding 결과 값 추출
  # 각 결과 값 label에 matching
  answer = np.argmax(result)
  if answer == 0:
    print("Label: 안투리움")
  elif answer == 1:
    print("Label: Ball Moss")
  elif answer == 2:
    print("Label: 참매발톱")
  elif answer == 3:
    print("Label: 가자니아")
  elif answer == 4:
    print("Label: 장미")
  elif answer == 5:
    print("Label: 해바라기")
  elif answer == 6:
    print("Label: wall flower")
  elif answer == 7:
    print("Label : yellow iris")
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
n6_t = 0
n6_f = 0
n7_t = 0
n7_f = 0


for i, ret in enumerate(os.walk('./test-data/anthurium')):
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
      n0_t = n0_t + 1
    else:
      n0_f = n0_t + 1

for i, ret in enumerate(os.walk('./test-data/ball moss')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      n1_t = n1_t + 1
    else:
      n1_f = n1_f + 1

for i, ret in enumerate(os.walk('./test-data/columbine')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      n2_t = n2_t + 1
    else:
      n2_f = n2_f + 1

for i, ret in enumerate(os.walk('./test-data/gazania')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 3:
      n3_t = n3_t + 1
    else:
      n3_f = n3_f + 1

for i, ret in enumerate(os.walk('./test-data/rose')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 4:
      n4_t = n4_t + 1
    else:
      n4_f = n4_f + 1

for i, ret in enumerate(os.walk('./test-data/sunflower')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 5:
      n5_t = n5_t +  1
    else:
      n5_f = n5_f + 1

for i, ret in enumerate(os.walk('./test-data/wallflower')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 6:
      n6_t = n6_t +  1
    else:
      n6_f = n6_f + 1

for i, ret in enumerate(os.walk('./test-data/yellow iris')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 7:
      n7_t = n7_t +  1
    else:
      n7_f = n7_f + 1

# """
# Check metrics
# """
print("True anthurium: ", n0_t)
print("False anthurium: ", n0_f)
print("True ball moss: ", n1_t)
print("False ball moss: ", n1_f)
print("True columbine: ", n2_t)
print("False columbine: ", n2_f)
print("True gazania: ", n3_t)
print("False gazania: ", n3_f)
print("True rose: ", n4_t)
print("False rose: ", n4_f)
print("True sunflower: ", n5_t)
print("False sunflower: ", n5_f)
print("True wallflower: ", n6_t)
print("False wallflower: ", n6_f)
print("True yellow iris: ", n7_t)
print("False yellow iris: ", n7_f)