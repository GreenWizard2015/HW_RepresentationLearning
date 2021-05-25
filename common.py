import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import Utils

def testLatentKMeansClassification(
  model,
  X_train, Y_train,
  X_test, Y_test,
  filepath
):
  N_CLASSES = 1 + Y_train.max()
  kmeans = KMeans(n_clusters=N_CLASSES)
  pred = kmeans.fit_predict(model.latent(X_train))
  
  clusterBySize = sorted(
    list(zip(*np.unique(pred, return_counts=True))),
    key=lambda x: x[1]
  )

  clusters2label = {}
  knownLabels = []
  for cluster, _ in clusterBySize:
    indices = np.where(pred == cluster)
    Ytrue = Y_train[indices]
    labels = dict(zip(*np.unique(Ytrue, return_counts=True)))
    for k in knownLabels:
      labels[k] = -1
      
    label = max(labels, key=labels.get)
    clusters2label[cluster] = label
    knownLabels.append(label)
  # test all data
  pred = kmeans.predict(model.latent(X_test))
  relabeled = np.zeros_like(pred)
  for c, l in clusters2label.items():
    relabeled[np.where(c == pred)] = l
  
  CM = confusion_matrix(Y_test, relabeled)
  Utils.plot_confusion_matrix(
    CM,
    target_names=[str(i) for i in range(N_CLASSES)],
    saveTo=filepath('confusion_matrix_kmeans.png'),
    onlyErrors=True,
    title='KMean'
  )
  return

def testClassificationByLatentAnchors(
  model, anchors, NClasses,
  X_test, Y_test,
  saveTo, title
):
  N = len(anchors) // NClasses
  pred_labels = np.zeros_like(Y_test)
  for i in range(len(X_test)):
    disimilarity = model.compareOne(anchors, X_test[i:i+1]).numpy()
    pred_labels[i] = disimilarity.argmin() // N
    
  CM = confusion_matrix(Y_test, pred_labels)
  Utils.plot_confusion_matrix(
    CM,
    target_names=[str(i) for i in range(NClasses)],
    saveTo=saveTo,
    onlyErrors=True,
    title=title
  )
  return

def meanLatentAnchors(model, X_train, Y_train):
  N_CLASSES = 1 + Y_train.max()
  anchors = []
  for label in range(N_CLASSES):
    indices = np.where(Y_train == label)[0]
    latents = model.latent(X_train[indices])
    meanLatent = latents.mean(axis=-2)
    anchors.append(meanLatent)
  return np.array(anchors)