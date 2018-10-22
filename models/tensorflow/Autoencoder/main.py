#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import *
from Autoencoder import *
import os

flags = tf.flags
PARAMETERS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 64, 'Batch size for both training and testing')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs in training phase')
flags.DEFINE_float('train_proportion', 0.8, 'Proportion of examples to consider for training only.')
flags.DEFINE_integer('feature_size', 2, 'Feature size learned by the network')
flags.DEFINE_string('loss_type', 'mse', 'Desired loss function to train (mse, bce)')

def main(args):
	#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        tf.reset_default_graph()
	tf.set_random_seed(1)
	np.random.seed(1)

	# load mnist data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	data = mnist.train.images
	labels = np.asarray(mnist.train.labels, dtype=np.int64)
	test_data = mnist.test.images
	test_labels = np.asarray(mnist.test.labels, dtype = np.int64)

	# hold out split - by default: 80% training and 20% validation
	train_data, train_labels, val_data, val_labels = partition_train_set(data, labels, PARAMETERS.train_proportion)

        # initialize autoencoder model
	ae = Autoencoder(PARAMETERS)

        # train the model with the mnist data
        history_loss = ae.train(train_data, val_data)

        # plot the history train/val loss
        plot_train_history_loss(history_loss)

        # test the model
        ae.test(test_data)
	
	# reconstruct some data
        test_batch_data = test_data[:10]
        reconstructed = ae.reconstruct_data(test_batch_data)  
        display_reconstructed(test_batch_data, reconstructed, 10)

        # plot the latent space
        fig = ae.plot_latent_space(test_data[:10000], test_labels[:10000])
        fig.show()

if __name__ == '__main__':
  tf.app.run()
