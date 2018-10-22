import tensorflow as tf

class LossFunctions:
    eps = 1e-8

    # p = real data, q = predicted data
    def binary_cross_entropy(self, p, q):
      loss = -tf.reduce_sum( p * tf.log(q + self.eps) + (1 - p) * tf.log(1 - q + self.eps), axis = 1 )
      return tf.reduce_mean(loss)

    def mean_square_error(self, p, q):
      loss = tf.square(p - q)
      return tf.reduce_mean(loss)
