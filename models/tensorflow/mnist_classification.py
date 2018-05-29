"""
MNIST classification using Tensorflow Dataset API
Author: Jhosimar Arias
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# define default hyperparameters
flags = tf.flags
PARAMETERS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_integer('batch_size', 50, 'Batch size for both training and testing')
flags.DEFINE_integer('num_classes', 10, 'Number of dataset classes')
flags.DEFINE_integer('image_size', 28, 'Image size, assuming same height and width')
flags.DEFINE_integer('num_channels', 1, 'Number of image channels')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs in training phase')


def resize_function(data, labels, params):
  """Data preprocessing

     - Reshape input data to the required format by tensorflow
     - Cast labels data type
  """
  data =  tf.reshape(data, [params.image_size, params.image_size, params.num_channels ])
  labels = tf.cast(labels, tf.int64)
  return data, labels


def create_dataset(is_training, data, labels, params):
  """Create dataset given input data
  
  Args:
      is_training: (bool) whether to use the train or test pipeline.
                   At training, we shuffle the data and have multiple epochs
      data: (array) corresponding array containing the input data
      labels: (list) corresponding list of labels
      params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

  Returns:
      output: (dict) contains what will be the input of the tensorflow graph
  """

  num_samples = data.shape[0]
  batch_size = params.batch_size
  num_classes = params.num_classes

  # create dataset object
  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.map(lambda _data, _labels: resize_function(_data, _labels, params))
  
  if is_training:  
    dataset = dataset.shuffle(num_samples).repeat().batch(batch_size).prefetch(1)
  else:
    dataset = dataset.batch(batch_size).prefetch(1)
  
  # create reinitializable iterator from dataset
  iterator = dataset.make_initializable_iterator()
  data, labels = iterator.get_next()
  iterator_init = iterator.initializer
  return {'data': data, 'labels': labels, 'iterator_init': iterator_init}


def build_model(is_training, inputs, params):
  """Compute logits of the model (output distribution)
  
  Args:
      is_training: (bool) whether we are training or not
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` or outputs of `tf.data`
      params: (Params) hyperparameters (ex: number of classes)
  
  Returns:
      logits: (tf.Tensor) output of the model
  """
  data = inputs['data']
  num_classes = params.num_classes
  filters = [32, 64]
  
  out = data
  # for each block, we do: 5x5 conv -> relu -> 2x2 maxpool
  for i, num_filters in enumerate(filters):
    # convolutional layer
    out = tf.layers.conv2d(out, num_filters, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
    # pooling layer
    out = tf.layers.max_pooling2d(out, pool_size=[2,2], strides=2)
  
  # fully connected layers
  out = tf.reshape(out, [-1, 7 * 7 * 64])
  out = tf.layers.dense(out, units=1024, activation=tf.nn.relu)
  out = tf.layers.dropout(out, rate=0.5, training=is_training)
  
  # output according to the number of classes/labels
  logits = tf.layers.dense(out, units=num_classes)
  
  return logits


def create_model(mode, inputs, params, reuse = False):
  """Model function defining the graph operations.
  
  Args:
      mode: (string) can be 'train' or 'test'
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` or outputs of `tf.data`
      params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
      reuse: (bool) whether to reuse the weights
  
  Returns:
      model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
  """
  is_training = (mode == 'train')
  labels = inputs['labels']
  
  with tf.variable_scope('model', reuse = reuse ):
    logits = build_model(is_training, inputs, params)
    prediction = tf.argmax(logits, axis=1)
  
  # sparse softmax considers labels as values between [0,num_classes]
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  accuracy = tf.reduce_mean( tf.cast( tf.equal(labels, prediction), tf.float32 ) )
  
  # optimize network only in training phase
  if is_training:
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss)
  
  model_spec = inputs
  model_spec['loss'] = loss
  model_spec['accuracy'] = accuracy
  model_spec['predictions'] = prediction
  model_spec['variable_init_op'] = tf.global_variables_initializer()
  
  if is_training:
    model_spec['train_op'] = train_op
  
  return model_spec
  

def evaluate_dataset(is_training, num_batches, model_spec, sess):
  """Evaluate the model
  
  Args:
      is_training: (bool) whether we are training or not
      num_batches: (integer) number of batches to train/test
      model_spec: (dict) contains the graph operations or nodes needed for evaluation
      sess: (tf.Session) current session

  Returns:
      avg_loss: (float) average loss for the given number of batches (training phase)
      avg_accuracy: (float) average accuracy for the given number of batches
  """
  avg_loss, avg_accuracy = 0, 0

  sess.run(model_spec['iterator_init'])
  if is_training:
    for j in range(num_batches):
      _, loss, accuracy = sess.run([model_spec['train_op'],
                                    model_spec['loss'], 
                                    model_spec['accuracy'] ])
      avg_loss = avg_loss + loss
      avg_accuracy = avg_accuracy + accuracy
  else:
    for j in range(num_batches):
      accuracy = sess.run(model_spec['accuracy'])
      avg_accuracy = avg_accuracy + accuracy

  avg_loss /= num_batches
  avg_accuracy /= num_batches
  
  return avg_loss, avg_accuracy


def main(args):
  # for reproducibility
  np.random.seed(1)
  tf.set_random_seed(1)

  # load mnist data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=False) 
  data = mnist.train.images
  labels = np.asarray(mnist.train.labels, dtype = np.int32)
  test_data = mnist.test.images
  test_labels = np.asarray(mnist.test.labels, dtype = np.int32)
  
  # hold out split - 80% training and 20% validation
  indices = np.random.permutation(data.shape[0])
  train_size = int(0.8 * data.shape[0])
  train_indices, val_indices = indices[:train_size], indices[train_size:]
  train_data, train_labels = data[train_indices], labels[train_indices]
  validation_data, validation_labels = data[val_indices], labels[val_indices]
  
  # create dataset objects and iterators
  train_dataset = create_dataset(True, train_data, train_labels, PARAMETERS)
  validation_dataset = create_dataset(False, validation_data, validation_labels, PARAMETERS)
  test_dataset = create_dataset(False, test_data, test_labels, PARAMETERS)
  
  # create train and test models
  train_model = create_model('train', train_dataset, PARAMETERS)
  validation_model = create_model('test', validation_dataset, PARAMETERS, True)
  test_model = create_model('test', test_dataset, PARAMETERS, True)
  
  # set number of batches
  TRAIN_N_BATCHES = train_data.shape[0] // PARAMETERS.batch_size
  VALIDATION_N_BATCHES = validation_data.shape[0] // PARAMETERS.batch_size
  TEST_N_BATCHES = test_data.shape[0] // PARAMETERS.batch_size
  
  with tf.Session() as sess:
    # initialize global variables
    sess.run( train_model['variable_init_op'] )
      
    # training and validation phases
    print('Training phase...')
    for i in range(PARAMETERS.num_epochs):
      print("Epoch %d" % (i + 1))

      # training
      loss, accuracy = evaluate_dataset(True, TRAIN_N_BATCHES, train_model, sess) 
      print("Loss: %lf" % loss)
      print("Training   Accuracy: %lf" % accuracy )

      #validation
      _, accuracy = evaluate_dataset(False, VALIDATION_N_BATCHES, validation_model, sess)  
      print("Validation Accuracy: %lf\n" % accuracy)
     
    # testing phase
    print('Testing phase...')
    _, accuracy = evaluate_dataset(False, TEST_N_BATCHES, test_model, sess)
    print("Testing    Accuracy: %lf\n" % accuracy)


if __name__ == '__main__':
    tf.app.run()
