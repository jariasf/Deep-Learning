"""
Sample code of the Tensorflow Dataset API
Author: Jhosimar Arias
"""

import tensorflow as tf
import numpy as np

def dataset_error():
    features = np.random.sample((10,2))
    labels = np.random.sample((10,1))
    # create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    # create one shot iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
      for i in range(11):
        try:
          print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
          print("OutOfRangeError while iterating")

          
def dataset_initializer():
    features = np.random.sample((10,2))
    labels = np.random.sample((10,1))
    # create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # create initializable iterator
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      for i in range(12):
        print(sess.run(next_element))
        # initialize iterator to avoid OutOfRangeError
        if i > 0 and i % 9 == 0:
          sess.run(iterator.initializer)
          
          
def dataset_batch():
    features = np.random.sample((10, 2))
    labels = np.random.sample((10,1))
    # create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(10).repeat().batch(3)
    # create one shot iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # set number of batches = number_elements/batch_size
    N_BATCHES = 10//3
    
    with tf.Session() as sess:
      for i in range(N_BATCHES):
        print("Batch %d" % i)
        print(sess.run(next_element))

        
def dataset_placeholder():
    features = np.random.sample((10,2))
    labels = np.random.sample((10,1))
    # create data placeholders
    x = tf.placeholder(tf.float32, shape=[None,2])
    y = tf.placeholder(tf.float32, shape=[None,1])
    #create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(10).repeat().batch(5)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # set number of batches = number_elements/batch_size
    N_BATCHES = 10//5
    
    with tf.Session() as sess:
      sess.run(iterator.initializer, feed_dict={x :features, y:labels })
      for i in range(N_BATCHES):
        print("Batch %d" % i)
        print(sess.run(next_element))
       
      
def dataset_reinitializable():
    #initialize train and validation values
    train_features = np.random.sample((10, 2))
    train_labels = np.random.sample((10,1))
    validation_features = np.random.sample((8,2))
    validation_labels = np.random.sample((8,1))
    #create dataset object
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
    train_dataset = train_dataset.shuffle(10).repeat().batch(5)
    validation_dataset = validation_dataset.repeat().batch(4)
    #create reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    train_iterator = iterator.make_initializer(train_dataset)
    validation_iterator = iterator.make_initializer(validation_dataset)
    # set number of batches = number_elements/batch_size
    TRAIN_N_BATCHES = 10 // 5
    VALIDATION_N_BATCHES = 8 // 4
    # set number of epochs
    EPOCHS = 5
    
    with tf.Session() as sess:
      # print batches values per epoch
      for i in range(EPOCHS):
        sess.run(train_iterator)
        print("Epoch: %d" % (i + 1))
        print("Train batches")
        for j in range( TRAIN_N_BATCHES ):
          print( sess.run(next_element))
    
        print("Validation batches")
        sess.run(validation_iterator)
        for j in range( VALIDATION_N_BATCHES):
          print( sess.run(next_element))


if __name__ == "__main__":
    np.random.seed(0)
    #dataset_error()
    #dataset_initializer()
    #dataset_batch()
    #dataset_placeholder()
    dataset_reinitializable()

