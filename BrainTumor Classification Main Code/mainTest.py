import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.models import Sequential
model=Sequential()

model = load_model('BrainTumor10EpochsCategorical.h5')
image=cv2.imread('/Users/d-tech/Downloads/Brain_Tumor_Classification-main/BrainTumor Classification DL/datasets/pred/pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

#result=model.predict_classes(input_img)

predict_x=model.predict(input_img) 
result=np.argmax(predict_x,axis=1)

print(result)




