# Solar Image Classification using Transfer Learning 

The objective of this project is to develop a deep learning model which can learn the different geometry of large-scale features on the Sun such that, after the model has been trained, a dataset of solar images can be passed to the network and it will identify which images contain which relevant feature in a very short time. 

Thus an image classification tool for identifying the large-scale features in a solar dataset has been built. It is based on the VGG-16 pretrained convolutional neural network (CNN).

## Dataset
The images in the dataset, were taken from Hinode/Solar Optical Telescope (SOT) using an Hα(6563 Å) filter([data available here](https://github.com/rhero12/Slic/releases/tag/1.1.1)). The different classification of the solar images are: filaments, flare ribbons, prominences, sunspots and the quiet Sun (i.e. the lack of any of the other four features).

Note: filaments and prominences are the same physical feature but are geometrically different so are treated as different classes here.

The model was trained for 5 epochs and a learning rate of η=0.0001 that reaches an training set accuracy of 99.95% and validation set accuracy of 100%.

## Requirements
**Libraries**
- TensorFlow
- Keras
- Python 3+
- numpy
- matplotlib
- scipy
- astropy
- scikit-image
- PIL

**System**
- 12 GB RAM(Minimum)
- Nvidia GPU


## Reference
["Fast Solar Image Classification Using Deep Learning and its Importance for Automation in Solar Physics](https://link.springer.com/article/10.1007/s11207-019-1473-z)", J.A. Armstrong and L. Fletcher, Solar Physics, vol. 294:80, (2019).
