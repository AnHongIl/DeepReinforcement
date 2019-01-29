import tensorflow as tf

class Network():
    def __init__(self, state_size, action_size, learning_rate, scope):        
        with tf.variable_scope(scope, reuse=False):        
            self.states = tf.placeholder(tf.float32, [None, *state_size], name='states')
            self.Ys = tf.placeholder(tf.float32, [None, 1], name='targetQ')

            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            self.one_hot_actions = tf.one_hot(self.actions, action_size)
        
            self.conv1 = tf.layers.conv2d(self.states, 16, [8, 8], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(self.conv1, [3, 3], 2)
            self.conv2 = tf.layers.conv2d(self.pool1, 32, [4, 4], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(self.conv2, [3, 3], 2)
            self.flatten = tf.layers.flatten(self.pool2)
            """
            self.conv3 = tf.layers.conv2d(self.pool2, 64, [2, 2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(self.conv3, [3, 3], 2)
            self.flatten = tf.layers.flatten(self.pool3)
            """
            #Too slow
            """
            self.conv1 = tf.layers.conv2d(self.states, 96, [11, 11], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(self.conv1, [3, 3], 2)
            
            self.conv2 = tf.layers.conv2d(self.pool1, 256, [5, 5], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(self.conv2, [3, 3], 2)
            
            self.conv3 = tf.layers.conv2d(self.pool2, 384, [3, 3], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.conv4 = tf.layers.conv2d(self.conv3, 384, [3, 3], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.conv5 = tf.layers.conv2d(self.conv4, 384, [3, 3], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(self.conv5, [3, 3], 2)
            
            self.flatten = tf.layers.flatten(self.pool3)                        
            """         
            self.hidden1 = tf.layers.dense(self.flatten, 512, tf.nn.relu, name='hidden1')
            self.hidden2 = tf.layers.dense(self.flatten, 512, tf.nn.relu, name='hidden2')
            
            self.V = tf.layers.dense(self.hidden1, 1, None, name="state_function")                        
            self.As = tf.layers.dense(self.hidden2, action_size, None, name="action_function")

            self.Qs = self.V + tf.subtract(self.As, tf.reduce_mean(self.As, axis=1, keepdims=True))

            self.Q = tf.reduce_sum(tf.multiply(self.Qs, self.one_hot_actions), axis=1, keepdims=True)

            self.loss = tf.reduce_mean(tf.square(self.Ys - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)        