import tensorflow as tf
import tensorflow.keras as keras

def _makeSiameseNetwork(model, latentDiff):
  shape = model.input_shape[1:]
  A = keras.Input(shape=shape)
  B = keras.Input(shape=shape)
  
  latentA = model(A)
  latentB = model(B)
  
  diff = keras.layers.Lambda(
    lambda x: latentDiff(x[0], x[1])
  )([latentA, latentB])
  
  return {'inputs': [A, B], 'outputs': [diff]}

def euclideanDistance(A, B):
  return tf.reduce_sum(tf.square(A - B), axis=-1)

class CSiameseNetwork(keras.Model):
  def __init__(self, model, alpha, comparator=euclideanDistance, **kwargs):
    kwargs.update(_makeSiameseNetwork(model, comparator))
    super().__init__(**kwargs)
    
    self._model = model
    self._comparator = comparator
    self._margin = alpha
    self.loss_tracker = keras.metrics.Mean(name="loss")
    return
  
  def latent(self, X):
    return self._model.predict([X])

  @tf.function
  def compareOne(self, anchors, sample):
    sample = self._model([sample])
    samples = tf.repeat(sample, tf.shape(anchors)[0], axis=0)
    return self._comparator(anchors, samples)
  
  def call(self, X, training=False):
    A, B = X
    lA = self._model([A], training=training)
    lB = self._model([B], training=training)
    return self._comparator(lA, lB)
  
  def _loss(self, data, training):
    anchors, positives, negatives = data
    lA = self._model([anchors], training=training)
    lP = self._model([positives], training=training)
    lN = self._model([negatives], training=training)
    
    posDist = self._comparator(lA, lP)
    negDist = self._comparator(lA, lN)
    
    return tf.maximum(0.0, (posDist - negDist) + self._margin)
    
  def train_step(self, data):
    with tf.GradientTape() as tape:
      loss = self._loss(data, training=True)
      
    TV = self.trainable_variables
    gradients = tape.gradient(loss, TV)
    self.optimizer.apply_gradients(zip(gradients, TV))
    
    self.loss_tracker.update_state(loss)
    return {"loss": self.loss_tracker.result()}
  
  def test_step(self, data):
    loss = self._loss(data, training=False)
    self.loss_tracker.update_state(loss)
    return {"loss": self.loss_tracker.result()}

  @property
  def metrics(self):
    return [self.loss_tracker]
