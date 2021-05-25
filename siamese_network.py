#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setup(MAX_GPU_MEMORY=4 * 1024, RANDOM_SEED=1671)

import Dataset
import tensorflow as tf
import numpy as np
import os
from CSiameseNetwork import CSiameseNetwork
import matplotlib.pyplot as plt
from CTripletsGenerator import CTripletsGenerator
from CPlotLatent import CPlotLatent
import shutil
import common

DATASET = 'mnist'
LATENT_ACTIVATION = 'linear'

NB_EPOCH = 100
BATCH_SIZE = 64
VERBOSE = 2
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = Dataset.dataset(DATASET, VALIDATION_SPLIT)
trainDataset = CTripletsGenerator(X_train, Y_train, batch_size=BATCH_SIZE)
valDataset = CTripletsGenerator(X_val, Y_val, batch_size=BATCH_SIZE)
N_CLASSES = 1 + Y_train.max()

for LATENT_SIZE in [8, 32, 128]:
  for LATENT_MARGIN in [0.0, 1.0, 4.0]:
    Utils.setupRandomSeed()
    FOLDER = os.path.join(
      os.path.dirname(__file__), 'output',
      'siamese-%s-%d-%.1f' % (DATASET, LATENT_SIZE, LATENT_MARGIN)
    )
    filepath = lambda *x: os.path.join(FOLDER, *x)
    shutil.rmtree(FOLDER, ignore_errors=True)
    os.makedirs(FOLDER, exist_ok=True)
    
    def createModel(shape, latentSize):
      L = tf.keras.layers
      res = data = L.Input(shape=shape)
      
      for filters in [16, 32, 64]:
        res = L.Conv2D(filters, 3, strides=2, activation='relu')(res)
        res = L.BatchNormalization()(res)
        
      res = L.Flatten()(res)
      for sz in [256, 128]:
        res = L.Dense(sz, activation='relu')(res)
    
      res = L.Dense(latentSize, activation=LATENT_ACTIVATION)(res)
      model = tf.keras.Model(inputs=[data], outputs=[res])
      model.summary()
      return model
    
    model = CSiameseNetwork(createModel(X_train.shape[1:], latentSize=LATENT_SIZE), LATENT_MARGIN)
    model.compile(tf.optimizers.Adam(learning_rate=1e-4))
    model.summary()
    
    def plot_triplets(anchors, positives, negatives):
      plt.clf()
      plt.figure(figsize=(2, 8))
      for i in range(len(anchors)):
        for j, img in enumerate([anchors, positives, negatives]):
          plt.subplot(len(anchors), 3, 3 * i + j + 1)
          plt.imshow(np.squeeze(img[i]))
          plt.xticks([])
          plt.yticks([])
    
      plt.savefig(filepath('triplets.png'), format='png')
      plt.close()
      return
    
    # debug
    a, p, n = trainDataset[0]
    plot_triplets(a[:15], p[:15], n[:15])
    
    # train
    history = model.fit(
      trainDataset,
      epochs=NB_EPOCH,
      verbose=VERBOSE,
      validation_data=valDataset, 
      callbacks=[
        tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', mode='min', patience=10,
          restore_best_weights=True
        ),
        CPlotLatent(
          model, X_test, Y_test,
          filepath=lambda epoch: filepath('latent-%s.png' % epoch)
        )
      ]
    ).history
    
    Utils.saveMetrics(history, lambda name: filepath(name))
    ######
    common.testLatentKMeansClassification(model, X_train, Y_train, X_test, Y_test, filepath)
    #########################
    # classify by selecting pairs
    def randomAnchors(N):
      anchors = np.zeros((N * N_CLASSES, *X_train.shape[1:]))
      for label in range(N_CLASSES):
        indices = np.random.choice( np.where(Y_train == label)[0], N )
        anchors[label * N:((label + 1) * N)] = X_train[indices]
      return model.latent(anchors)
    
    for K in [1, 5, 10]:
      print('Test %d-shots classification' % K)
      anchors = randomAnchors(K)
      common.testClassificationByLatentAnchors(
        model, anchors, N_CLASSES,
        X_test, Y_test,
        saveTo=filepath('confusion_matrix_kshots-%d.png' % K),
        title='K = %d' % K
      )
      
    print('Test classification by mean anchors')
    anchors = common.meanLatentAnchors(model, X_train, Y_train)
    common.testClassificationByLatentAnchors(
      model, anchors, N_CLASSES,
      X_test, Y_test,
      saveTo=filepath('confusion_matrix_mean.png'),
      title='Mean anchors'
    )