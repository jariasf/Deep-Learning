import numpy as np
import matplotlib.pyplot as plt

def del_all_flags(FLAGS):
  flags_dict = FLAGS._flags()
  keys_list = [keys for keys in flags_dict]
  for keys in keys_list:
    FLAGS.__delattr__(keys)


def partition_train_set(data, labels, proportion):
  num_samples = data.shape[0]
  indices = np.random.permutation(num_samples)
  train_size = int(proportion * num_samples)
  train_indices, val_indices = indices[:train_size], indices[train_size:]
  train_data, train_labels = data[train_indices], labels[train_indices]
  val_data, val_labels = data[val_indices], labels[val_indices]
  return train_data, train_labels, val_data, val_labels


def plot_train_history_loss(history):
  # summarize history for loss
  plt.plot(history['train_history_loss'])
  plt.plot(history['val_history_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper right')
  plt.show()

def display_reconstructed(original, reconstructed, n=10):
  plt.figure(figsize=[20,4])
  for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(original[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    
    if reconstructed is not None:
      plt.subplot(2, n, i + n + 1)
      plt.imshow(reconstructed[i].reshape(28, 28))
      plt.gray()
      plt.axis('off')
  plt.show()
