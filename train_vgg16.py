
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Flatten

train_file = "/content/drive/My Drive/CL_398/solar_train_data/data.npy" #file location
train_data = np.load(train_file) #return the input array from a disk file with npy extension(.npy)
print(train_data.shape)

val_file = "/content/drive/My Drive/CL_398/solar_test_data/data.npy"
val_data = np.load(val_file)
print(val_data.shape)


img=train_data[1,1:].reshape( 256, 256)
image = np.dstack([img,img,img])
train_label =  train_data[:, 0 ]
val_label=val_data[:,0]
# print(train_label)
# print(val_label)

img=train_data[3000,1:].reshape( 256, 256)

image = np.dstack([img,img,img])

# plt.figure() 
# plt.imshow(image) #used to visualise the image
# print(image.shape)

train_data=train_data[:, 1: ]
train_data.shape



train_img = []
for i in range(train_data.shape[0]):
  img=train_data[i,0:].reshape(256,256)
  image= np.dstack([img,img,img]) 
  train_img.append(image)

train_img = np.array(train_img)
# print(train_img.shape,train_label.shape)

val_img = []
val_data=val_data[:, 1: ]
for i in range(val_data.shape[0]):
  img=val_data[i,0:].reshape(256,256)
  image= np.dstack([img,img,img]) 
  val_img.append(image)

val_img = np.array(val_img)
# print(val_img.shape,val_label.shape)



"""##Using VGG-16 Pre-trained Model"""
# load model without output layer
base_model = VGG16(include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False
flat1 = Flatten()(base_model.layers[-1].output)
drop1 = Dropout(0.25)(flat1)
class1 = Dense(1024, activation='relu')(drop1)
drop2 = Dropout(0.25)(class1)
output = Dense(5, activation='softmax')(drop2)
# define new model
model = Model(inputs=base_model.inputs, outputs=output)

# print(model.summary())

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='vgg.png')

from sklearn.utils import shuffle
train_img, train_label = shuffle(train_img, train_label,random_state=32)

val_img, val_label = shuffle(val_img, val_label,random_state=25)

print(train_img.shape,train_label.shape)
print(val_img.shape,val_label.shape)

"""**Model Training**"""

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
num_classes = 5
batch=64

opt = RMSprop(lr=0.001)
model.compile(optimizer=opt,  # 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

History=model.fit(train_img, train_label,
          epochs=10,
          batch_size=batch,
          validation_data=(val_img, val_label),
          verbose=1)

model.save("/content/drive/My Drive/CL_398/Solar_VGG16_RMS(0.001).h")
print("Model Saved")

##Saving model data
import pandas as pd
hist_df = pd.DataFrame(History.history) 
hist_csv_file = '/content/drive/My Drive/CL_398/history_VGG16_RMS(0.001).csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

hist_df.head()

