import tensorflow.keras.datasets as DS

def dataset(name, VALIDATION_SPLIT, dtype='float32'):
  DATASETS = {
    'mnist': DS.mnist,
    'cifar10': DS.cifar10,
    'cifar100': DS.cifar100,
    'fashion_mnist': DS.fashion_mnist,
  }
  (X_train, Y_train), (X_test, Y_test) = DATASETS[name].load_data()
  if 3 == len(X_train.shape): # fix mnist
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
  # normalize
  X_train = X_train.astype(dtype) / 255.0
  X_test = X_test.astype(dtype) / 255.0
  # split train into validation and train
  splitIndex = int(len(X_train) * VALIDATION_SPLIT)
  X_val = X_train[:splitIndex]
  Y_val = Y_train[:splitIndex]
  
  X_train = X_train[splitIndex:]
  Y_train = Y_train[splitIndex:]

  return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
