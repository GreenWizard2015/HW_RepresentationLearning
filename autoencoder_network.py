#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setup(MAX_GPU_MEMORY=4 * 1024, RANDOM_SEED=1671)

import Dataset
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from CAutoencoderNetwork import CAutoencoderNetwork
from CPlotLatent import CPlotLatent
from CAEPairsGenerator import CAEPairsGenerator
import shutil
import common

DATASET = 'mnist'

NB_EPOCH = 1000
BATCH_SIZE = 64
VERBOSE = 2
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = Dataset.dataset(DATASET, VALIDATION_SPLIT)
N_CLASSES = 1 + Y_train.max()

def createEncoder(latentSize, latentActivation, VAE=False):
  L = tf.keras.layers
  res = data = L.Input(shape=X_train.shape[1:])
  
  for filters in [16, 32, 64]:
    res = L.Conv2D(filters, 3, padding='same', activation=None)(res)
    res = L.LeakyReLU(-0.1)(res)
    res = L.Conv2D(filters, 3, strides=2, activation=None)(res)
    res = L.LeakyReLU(-0.1)(res)
    
  res = L.Flatten()(res)
  for sz in [256, 128]:
    res = L.Dense(sz, activation=None)(res)
    res = L.LeakyReLU(-0.1)(res)

  outputs = None
  if VAE:
    def sample(inputs):
      z_mean, z_log_var = inputs
      noise = tf.random.normal(shape=tf.shape(z_mean))
      return z_mean + tf.exp(0.5 * z_log_var) * noise
    #####
    mean = L.Dense(latentSize, activation=latentActivation)(res)
    logVar = L.Dense(latentSize, activation='linear')(res)
    
    outputs = [L.Lambda(sample)([mean, logVar])]
  else:
    outputs = [ L.Dense(latentSize, activation=latentActivation)(res) ]
  
  model = tf.keras.Model(inputs=[data], outputs=outputs)
  model.summary()
  return model

def createDecoder(latentSize):
  L = tf.keras.layers
  res = data = L.Input(shape=(latentSize,))
  
  H, W, D = X_train.shape[1:]
  assert H == W, 'Must have same height and width'
  assert 0 == (H % 4), 'Height/width must be a multiple of 4'
  
  minSize = (H // 4) * (W // 4)
  res = L.Dense(minSize, activation=None)(res)
  res = L.LeakyReLU(-0.1)(res)
  res = L.Reshape((H // 4, W // 4, 1))(res)
  
  res = L.Conv2D(4, 3, activation=None, padding='same')(res)
  res = L.LeakyReLU(-0.1)(res)
  res = L.Conv2DTranspose(8, 2, strides=2, activation=None, padding='same')(res)
  res = L.LeakyReLU(-0.1)(res)
  
  res = L.Conv2D(16, 3, activation=None, padding='same')(res)
  res = L.LeakyReLU(-0.1)(res)
  res = L.Conv2DTranspose(32, 2, strides=2, activation=None, padding='same')(res)
  res = L.LeakyReLU(-0.1)(res)
  
  res = L.Conv2D(32, 3, activation=None, padding='same')(res)
  res = L.LeakyReLU(-0.1)(res)
  
  res = L.Conv2D(D, 3, activation=None, padding='same')(res)
  model = tf.keras.Model(inputs=[data], outputs=[res])
  model.summary()
  return model

def trainAndTest(LATENT_SIZE, NOISE, USE_VAE, LATENT_ACTIVATION):
  Utils.setupRandomSeed()
  modelName = 'AE-%s-%s-%s-%d-%.2f' % (
    DATASET, 'V' if USE_VAE else 'N', LATENT_ACTIVATION, LATENT_SIZE, NOISE
  )
  print('Start training %s' % modelName)
  FOLDER = os.path.join(os.path.dirname(__file__), 'output', modelName)
  filepath = lambda *x: os.path.join(FOLDER, *x)
  shutil.rmtree(FOLDER, ignore_errors=True)
  os.makedirs(FOLDER, exist_ok=True)
  
  model = CAutoencoderNetwork(
    encoder=createEncoder(LATENT_SIZE, LATENT_ACTIVATION, VAE=USE_VAE),
    decoder=createDecoder(LATENT_SIZE)
  )
  model.compile(tf.optimizers.Adam(learning_rate=1e-3), loss='mse')
  model.summary()

  # train
  history = model.fit(
    CAEPairsGenerator(X_train, BATCH_SIZE, noise=NOISE),
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=CAEPairsGenerator(X_val, BATCH_SIZE, noise=0.0), 
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=5,
        restore_best_weights=True
      ),
      CPlotLatent(
        model, X_test, Y_test,
        filepath=lambda epoch: filepath('latent-%s.png' % epoch),
        repeats=5 if USE_VAE else 1
      )
    ]
  ).history
  
  Utils.saveMetrics(history, lambda name: filepath(name))
  ######
  common.testLatentKMeansClassification(model, X_train, Y_train, X_test, Y_test, filepath)
  #########################
  # classify by selecting pairs
  print('Test classification by mean anchors')
  anchors = common.meanLatentAnchors(model, X_train, Y_train)
  common.testClassificationByLatentAnchors(
    model, anchors, N_CLASSES,
    X_test, Y_test,
    saveTo=filepath('confusion_matrix_mean.png'),
    title='Mean anchors'
  )
  ##################
  # test denoising
  for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    Utils.setupRandomSeed()
    gen = CAEPairsGenerator(X_test, 10, noise)
    noised, samples = gen[0]
    
    images = [noised]
    images.append(model([noised]).numpy())
    images.append(samples)
    plt.clf()
    plt.figure(figsize=(2/3 * len(images), 6))
    for i in range(len(samples)):
      for j, img in enumerate(images):
        plt.subplot(len(samples), len(images), len(images) * i + j + 1)
        plt.imshow(np.squeeze(img[i]))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(filepath('denoise-%.2f.png' % (noise, )), format='png')
    plt.close()
    continue
  return

for LATENT_ACTIVATION in ['linear', 'tanh', 'sigmoid']:
  for NOISE in [0.0, 0.1, 0.2, 0.3]:
    for USE_VAE in [True, False]:
      for LATENT_SIZE in [8, 32, 128]:
        trainAndTest(LATENT_SIZE, NOISE, USE_VAE, LATENT_ACTIVATION)