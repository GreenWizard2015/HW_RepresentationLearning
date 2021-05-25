import tensorflow as tf
import tensorflow.keras as keras

def _makeAutoencoderNetwork(encoder, decoder):
  shape = encoder.input_shape[1:]
  A = keras.Input(shape=shape)
  
  latent = encoder(A)
  decoded = decoder(latent)

  return {'inputs': [A], 'outputs': [decoded]}

class CAutoencoderNetwork(keras.Model):
  def __init__(self, encoder, decoder, **kwargs):
    kwargs.update(_makeAutoencoderNetwork(encoder, decoder))
    super().__init__(**kwargs)
    
    self._encoder = encoder
    self._decoder = decoder
    return
  
  def latent(self, X):
    return self._encoder.predict(X)
  
  def decode(self, X):
    return self._decoder.predict(X)

  @tf.function
  def compareOne(self, anchors, sample):
    sample = self._encoder(sample)
    samples = tf.repeat(sample, tf.shape(anchors)[0], axis=0)
    return tf.losses.MeanSquaredError(tf.losses.Reduction.NONE)(anchors, samples)