import tensorflow as tf

class AutoencoderNetwork:

    def encoder(self, input_data, feature_size, is_training=False):    
      out = input_data
      out = tf.layers.dense(out, units=512, activation=tf.nn.relu)
      out = tf.layers.dense(out, units=256, activation=tf.nn.relu)
      out = tf.layers.dense(out, units=feature_size)
      return out

    def decoder(self, features, output_size, is_training=False):
      out = features
      out = tf.layers.dense(out, units=256, activation=tf.nn.relu)
      out = tf.layers.dense(out, units=512, activation=tf.nn.relu)
      out = tf.layers.dense(out, units=output_size, activation=tf.nn.sigmoid)
      return out
    
    def build_model(self, input_data, feature_size, output_size):
      features = self.encoder(input_data, feature_size)
      output = self.decoder(features, output_size)
      return output
