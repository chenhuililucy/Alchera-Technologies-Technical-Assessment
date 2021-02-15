# your implementation goes here


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import os 
from PIL import Image
import pickle


#dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)

def readimage(): 

    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """


    img_path="/Users/lichenhuilucy/Desktop/rgb.png"
    img_groundtruth="/Users/lichenhuilucy/Desktop/gt.png"

    #Read in images
    image=cv2.imread(img_path,3)
    groundtruth=cv2.imread(img_groundtruth,0)

    # Resize images
    dsize=(256,256)
    image=cv2.resize(image,dsize)
    groundtruth= np.dstack((groundtruth, groundtruth))  
    groundtruth=cv2.resize(groundtruth,dsize)
    img = Image.fromarray(groundtruth)
    img.show()
    img = Image.fromarray(image)
    img.show()
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    groundtruth=groundtruth.reshape(1,groundtruth.shape[0],groundtruth.shape[1],groundtruth.shape[2])
    groundtruth = tf.cast(groundtruth, tf.float32) / 255.0
    #groundtruth = tf.cast(groundtruth, tf.float32) 
    image = tf.cast(image, tf.float32)/ 255.0
    #image = tf.cast(image, tf.float32)
    return image.shape,groundtruth.shape, image, groundtruth


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > 0.995):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(0.995*100))
			self.model.stop_training = True


# # Instantiate a callback object
# #callbacks = myCallback()
# def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
#     ''' 
#     Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
#     Assumes the `channels_last` format.
  
#     # Arguments
#         y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
#         y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
#         epsilon: Used for numerical stability to avoid divide by zero errors
    
#     # References
#         V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
#         https://arxiv.org/abs/1606.04797
#         More details on Dice loss formulation 
#         https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
#         Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
#     '''
    
#     # skip the batch and class axis for calculating Dice score


#     axes = tuple(range(1, len(y_pred.shape)-1)) 
#     numerator = 2. * tf.experimental.numpy.sum((y_pred * y_true, axes))
#     denominator = tf.experimental.numpy.sum(tf.experimental.numpy.square(y_pred) + tf.experimental.numpy.square(y_true), axes)
    
#     return 1 - tf.experimental.numpy.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
#     # thanks @mfernezir for catching a bug in an earlier version of this implementation!



def dice_loss(softmax_output, labels, ignore_background=False, square=False):
    if ignore_background:
      labels = labels[..., 1:]
      softmax_output = softmax_output[..., 1:]
    axis = (0,1,2)
    eps = 1e-7
    nom = (2 * tf.reduce_sum(labels * softmax_output, axis=axis) + eps)
    if square:
      labels = tf.square(labels)
      softmax_output = tf.square(softmax_output)
    denom = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum(softmax_output, axis=axis) + eps
    return 1 - tf.reduce_mean(nom / denom)


#callback=myCallback()

def train():
    imageshape, groundtruthshape, image, groundtruth=readimage()
    print(imageshape)
    print(groundtruthshape)
    model=tf.keras.Sequential()
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(imageshape[0],imageshape[1],3)))
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(tf.keras.layers.Conv2D(2,kernel_size=(5,5)))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #lossf=soft_dice_loss(groundtruth,history)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    history=model.fit(image,groundtruth,epochs=100,callbacks=[myCallback()])
    print(history)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('loss_vs_epochs.png')
    #pickle.dump(model,open(filename,"wb"))
    model.save('finalized_model.h5')
    pred_mask=model.predict(image)
    pred_mask.resize(256,256)
    img = Image.fromarray(pred_mask)
    img.show()
    plt.imshow(pred_mask,cmap="gray_r")
    plt.show()

if __name__ == '__main__':
    history=train() 
    #plot()