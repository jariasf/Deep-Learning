import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from FullyConnected import *
from LossFunctions import * 

class Autoencoder:

    def __init__(self, params):
      self.learning_rate = params.learning_rate
      self.batch_size = params.batch_size
      self.feature_size = params.feature_size
      self.num_epochs = params.num_epochs
      self.loss_type = params.loss_type
      self.sess = tf.Session()
      self.network = AutoencoderNetwork()
      self.losses = LossFunctions()
    
    
    def create_dataset(self, is_training, data, batch_size):
      """Create dataset given input data

      Args:
          is_training: (bool) whether to use the train or test pipeline.
                       At training, we shuffle the data and have multiple epochs
          data: (array) corresponding array containing the input data
          batch_size: (int) size of each batch to consider from the data
 
      Returns:
          output: (dict) contains what will be the input of the tensorflow graph
      """
      num_samples = data.shape[0]

      # create dataset object
      dataset = tf.data.Dataset.from_tensor_slices(data)

      if is_training:  
        dataset = dataset.shuffle(num_samples).repeat()

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(1)

      # create reinitializable iterator from dataset
      iterator = dataset.make_initializable_iterator()
      data = iterator.get_next()
      iterator_init = iterator.initializer
      return {'data': data, 'iterator_init': iterator_init}
       

    def create_model(self, is_training, inputs, output_size):
      """Model function defining the graph operations.

      Args:
          is_training: (bool) whether we are training or not
          inputs: (dict) contains the inputs of the graph (features, labels...)
                  this can be `tf.placeholder` or outputs of `tf.data`
          output_size: (int) size of the output layer

      Returns:
          model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
      """
      data = inputs['data']

      with tf.variable_scope('model', reuse=not is_training):
        features = self.network.encoder(data, self.feature_size, is_training)
        output = self.network.decoder(features, output_size, is_training)
    
      if self.loss_type == 'bce':
        loss = self.losses.binary_cross_entropy(data, output)
      else:
        loss = self.losses.mean_square_error(data, output)
      
      if is_training:
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)
      
      model_spec = inputs
      model_spec['loss'] = loss
      model_spec['variable_init_op'] = tf.global_variables_initializer()
      model_spec['features'] = features
      model_spec['output'] = output
      
      if is_training:
        model_spec['train_op'] = train_op

      return model_spec
    

    def evaluate_dataset(self, is_training, num_batches, model_spec):
      """Evaluate the model

      Args:
          is_training: (bool) whether we are training or not
          num_batches: (integer) number of batches to train/test
          model_spec: (dict) contains the graph operations or nodes needed for evaluation

      Returns:
          avg_loss: (float) average loss for the given number of batches (training phase)
      """
      avg_loss = 0.0
      self.sess.run(model_spec['iterator_init'])
      if is_training:
        for j in range(num_batches):
          _, loss = self.sess.run([model_spec['train_op'], model_spec['loss'] ])
          avg_loss = avg_loss + loss
      else:
        for j in range(num_batches):
          loss = self.sess.run(model_spec['loss'])
          avg_loss = avg_loss + loss

      avg_loss /= num_batches
      return avg_loss

    
    def train(self, train_data, val_data):
      """Train the model

      Args:
          train_data: (array) corresponding array containing the training data
          val_data: (array) corresponding array containing the validation data

      Returns:
          output: (dict) contains the history of train/val loss
      """
      train_history_loss = []
      val_history_loss = []
      
      # create training and validation dataset
      train_dataset = self.create_dataset(True, train_data, self.batch_size)
      val_dataset = self.create_dataset(False, val_data, self.batch_size)
      
      # create train and validation models
      output_size = train_data.shape[1]
      train_model = self.create_model(True, train_dataset, output_size)
      val_model = self.create_model(False, val_dataset, output_size)
    
      # set number of batches
      TRAIN_BATCHES = train_data.shape[0] // self.batch_size
      VALIDATION_BATCHES = val_data.shape[0] // self.batch_size
      
      # initialize global variables
      self.sess.run( train_model['variable_init_op'] )

      # training and validation phases
      print('Training phase...')
      for i in range(self.num_epochs):
        train_loss = self.evaluate_dataset(True, TRAIN_BATCHES, train_model) 
        val_loss = self.evaluate_dataset(False, VALIDATION_BATCHES, val_model)  

        print("(Epoch %d / %d) Training Loss: %.5lf; Validation Loss: %.5lf" % \
              (i + 1, self.num_epochs, train_loss, val_loss))
        
        train_history_loss.append(train_loss)
        val_history_loss.append(val_loss)
        
      return {'train_history_loss' : train_history_loss,
              'val_history_loss': val_history_loss}
    
    
    def test(self, test_data):
      """Test the model

      Args:
          test_data: (array) corresponding array containing the testing data

      """
      # create dataset
      test_dataset = self.create_dataset(False, test_data, self.batch_size)

      # reuse model used in training phase
      output_size = test_data.shape[1]
      test_model = self.create_model(False, test_dataset, output_size)
      
      # evaluate model 
      TEST_BATCHES = test_data.shape[0] // self.batch_size
      print("Testing phase...")
      test_loss = self.evaluate_dataset(False, TEST_BATCHES, test_model)
      print("Testing Loss: %lf\n" % test_loss)
    

    def latent_features(self, data, batch_size=-1):
      """Obtain latent features learnt by the model

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          features: (array) array containing the features from the data
      """
      if batch_size == -1:
        batch_size = data.shape[0]
      
      # create dataset  
      dataset = self.create_dataset(False, data, batch_size)
      
      # we will use only the encoder network
      with tf.variable_scope('model', reuse=True):      
        encoder = self.network.encoder(dataset['data'], self.feature_size)
      
      # obtain the features from the input data
      self.sess.run(dataset['iterator_init'])      
      num_batches = data.shape[0] // batch_size
      features = np.zeros((data.shape[0], self.feature_size))
      for j in range(num_batches):
        features[j*batch_size:j*batch_size + batch_size] = self.sess.run(encoder)
      return features
    
    
    def reconstruct_data(self, data, batch_size=-1):
      """Reconstruct Data

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          reconstructed: (array) array containing the reconstructed data
      """
      if batch_size == -1:
        batch_size = data.shape[0]

      # create dataset
      dataset = self.create_dataset(False, data, batch_size)

      # reuse model used in training
      model_spec = self.create_model(False, dataset, data.shape[1])

      # obtain the reconstructed data
      self.sess.run(model_spec['iterator_init'])      
      num_batches = data.shape[0] // batch_size      
      reconstructed = np.zeros(data.shape)
      pos = 0
      for j in range(num_batches):
        reconstructed[pos:pos + batch_size] = self.sess.run(model_spec['output'])
        pos += batch_size
      return reconstructed
    

    def plot_latent_space(self, data, labels, save=False):
      """Plot the latent space learnt by the model

      Args:
          data: (array) corresponding array containing the data
          labels: (array) corresponding array containing the labels
          save: (bool) whether to save the latent space plot

      Returns:
          fig: (figure) plot of the latent space
      """
      # obtain the latent features
      features = self.latent_features(data)
      
      # plot only the first 2 dimensions
      fig = plt.figure(figsize=(8, 6))
      plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
              edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
      plt.colorbar()
      if(save):
          fig.savefig('latent_space.png')
      return fig
