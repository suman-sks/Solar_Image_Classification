from keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
model = load_model('/content/drive/My Drive/CL_398/Solar_VGG16_Adam(0.001).h')
opt = Adam(lr=0.0001)
model.compile(optimizer=opt,  # 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

from keras.preprocessing.image import img_to_array,load_img
import numpy as np
from skimage.transform import resize
import cv2
import os

images = []

image = load_img('/content/drive/MyDrive/CL_398/Test_data/p2.jpg', target_size=(256, 256))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)


classes = model.predict(image)

print(classes)
class_labels = ['filaments', 'flares', 'prominences', 'quiet', 'sunspots']
class_pred=[]
for i in range(len(classes)):
 class_pred.append(class_labels[np.argmax(classes[i])])

# classes=list(classes)
classes.shape

import matplotlib.pyplot as plt
plt.bar(class_labels, classes[0,:])
plt.xticks(class_labels)

plt.xlabel('Class_Labels')
plt.ylabel('Probability')

scores = model.evaluate(val_img, val_label, verbose=0)

import sklearn.metrics as metrics

y_pred_ohe = model.predict(val_img)  # shape=(n_samples, 12)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

confusion_matrix = metrics.confusion_matrix(y_true=val_label, y_pred=y_pred_labels)  # shape=(12, 12)

confusion_matrix

plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

