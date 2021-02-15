# your implementation goes here

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import pickle
from keras.models import load_model
from IPython.display import display
from PIL import Image

def readimage(): 
    img_path="/Users/lichenhuilucy/Desktop/rgb.png"
    image=cv2.imread(img_path,3)
    # Resize images
    dsize=(256,256)
    image=cv2.resize(image,dsize)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image = tf.cast(image, tf.float32)
    return image

def loadmodel(): 
    filename = 'finalized_model.h5'
    model = load_model(filename)
    return model


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False



def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
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
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show(ds, num=2):
    for data, _ in ds.take(num):
        plt.imshow(data[0, :, :, :].squeeze(), cmap=plt.cm.gray_r)
        plt.show()

def show_predictions(): 
    image=readimage()
    model=loadmodel()
    inference=model.predict(image)
    pred_mask = create_mask(inference)
    #pred_mask = pred_mask.numpy()
    plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]),cmap="gray")
    plt.show()
    pred_mask.resize(256,256)
    img = Image.fromarray(pred_mask)
    img.show()
    img.savefig('predictions.jpg')

    

if __name__ == '__main__':
    show_predictions()
    