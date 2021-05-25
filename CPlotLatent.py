import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

class CPlotLatent(tf.keras.callbacks.Callback):
  def __init__(self, model, X, Y, filepath, saveFreq=-1, repeats=1, pca=True):
    super().__init__()
    self._model = model
    self._X = X
    self._Y = np.squeeze(Y)
    self._filepath = filepath
    self._best = float('inf')
    self._saveFreq = saveFreq
    self._repeats = repeats
    self._pca = pca
    return

  def on_epoch_end(self, epoch, logs=None):
    loss = logs['val_loss']
    scheduled = (0 < self._saveFreq) and (0 == (epoch % self._saveFreq))
    if (self._best < loss) and not scheduled: return
    
    self._best = min((self._best, loss))
    self.dump('%06d' % epoch)
    return
  
  def dump(self, epoch):
    dest = self._filepath(epoch)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    plt.clf()
    plt.figure(figsize=(16, 16))
    plt.title('Epoch: ' + epoch)
    cmap = plt.cm.get_cmap('gist_rainbow', 10)
    
    for _ in range(self._repeats):
      projected = self._model.latent(self._X)
      if self._pca:
        pca = PCA(2) # project to 2 dimensions
        projected = pca.fit_transform(projected)
      
      plt.scatter(
        projected[:, 0], projected[:, 1], c=self._Y,
        edgecolor='none', alpha=0.5, cmap=cmap
      )
    ########
    # only last iteration
    for label in range(1 + self._Y.max()):
      (indices,) = np.where(label == self._Y)
      pts = projected[indices]
      cx = pts[:, 0].sum() / len(pts)
      cy = pts[:, 1].sum() / len(pts)
      plt.plot([cx], [cy], 'o', color='black')
      plt.text(cx * (1 + 0.01), cy * (1 + 0.01), label, fontsize=24)
    ########
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    
    plt.savefig(dest)
    plt.close()
    return