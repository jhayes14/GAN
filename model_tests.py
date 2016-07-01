from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

import pytest
import glob
import os
import train
import cv2
import model

# ----------- TESTS -------------

def test_generator_model():
    epochs = 1
    input_data = np.random.rand(1, 100)
    input_shape = input_data.shape
    generator = model.generator_model()
    adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    pred = generator.predict(input_data)
    return pred.shape

def test_discriminator_model():
    epochs = 1
    input_data = np.random.rand(1, 3, 64, 64)
    input_shape = input_data.shape

    discriminator = model.discriminator_model()
    adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    pred = discriminator.predict(input_data)
    print pred

def test_encoder_model():
    epochs = 1
    input_data = np.random.rand(5,5)
    input_shape = input_data.shape

    enc, vae = model.vaencoder_model()
    adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    enc.compile(loss='binary_crossentropy', optimizer=adam)
    pred = enc.predict(input_data)
    print pred

def test_vae_model():
    epochs = 1
    input_data = np.random.rand(5,5)
    input_shape = input_data.shape

    enc, vae = model.vaencoder_model()
    adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    vae.compile(loss='binary_crossentropy', optimizer=adam)
    pred = vae.predict(input_data)
    print pred

def test_train_discriminator():
    """
    Make sure that the discriminator can achieve low loss, when
    not training the generator.
    """
    path = r'../fauxtograph/images/'
    paths = glob.glob(os.path.join(path, "*.jpg"))
    # Load images
    real_images = np.array( [ train.load_image(p) for p in paths ] )
    np.random.shuffle( real_images )
    total_samples, c_dim, x_dim, y_dim = real_images.shape

    train_real_images = np.array( [ im for im in real_images[ : int(total_samples/2)] ] )
    test_real_images = np.array( [ im for im in real_images[int(total_samples/2) : ] ] )

    fake_images = np.array( [ np.random.uniform(-1, 1, (3,64,64)) for n in range(len(real_images)) ] )

    train_fake_images = np.array( [ im for im in fake_images[ : int(total_samples/2)] ] )
    test_fake_images = np.array( [ im for im in fake_images[int(total_samples/2) : ] ] )

    assert len(train_fake_images) == len(train_real_images)
    assert len(test_fake_images) == len(test_real_images)

    X_train = np.concatenate((train_real_images, train_fake_images))
    y_train = [1] * len(train_real_images) + [0] * len(train_fake_images) # labels

    X_test = np.concatenate((test_real_images, test_fake_images))
    y_test = [1] * len(test_real_images) + [0] * len(test_fake_images) # labels

    discriminator = model.discriminator_model()
    adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    discriminator.fit(X_train, y_train, batch_size=128, nb_epoch=2, verbose=1, validation_data=(X_test, y_test) )

def test_random_image():
    """
        Check the image made by np.random.uniform does indeed look noisy.
    """
    R = np.random.uniform(-1, 1, (64,64))
    G = np.random.uniform(-1, 1, (64,64))
    B = np.random.uniform(-1, 1, (64,64))

    img = np.array( [R, G, B] )

    rolled = np.rollaxis(img, 0, 3)
    cv2.imwrite('results/TEST.jpg', np.uint8(255 * 0.5 * (rolled + 1.0)))

def test_check_gen_model():
    '''
        Check generator creates correct image size
    '''
    generator = model.generator_model()
    adam_gen=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    #fake = train.noise_image()
    fake = np.array( [ train.noise_image() for n in range(1) ] )
    fake_predit = generator.predict(fake)

    rolled = np.rollaxis(fake_predit[0], 0, 3)
    print rolled
    #print np.uint8(255 * 0.5 * (rolled + 1.0))

def test_vis_img():
    '''
        Visualize real image quality when compressed to 64x64.
        This gives a base of how good the generator is.
    '''
    path = r'/Users/jamie/Downloads/101_ObjectCategories/Faces_easy/image_0080.jpg'
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (64, 64))) / 127.5 - 1
    img = np.rollaxis(img, 2, 0)

    rolled = np.rollaxis(img, 0, 3)
    cv2.imwrite('results/real_compressed_face.jpg', np.uint8(255 * 0.5 * (rolled + 1.0)))


if __name__ == "__main__":

    test_vis_img()
    pass
