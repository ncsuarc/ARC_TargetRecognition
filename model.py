import tensorflow as tf

class Model:
    def __init__(self, n_input, n_classes, dropout):
        self.n_input = n_input
        self.n_classes = n_classes
        self.dropout = dropout
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        self.predictor = self.conv_net()
    
    def conv_net(self):
        #Reshape input image
        x_image = tf.reshape(self.x, shape=[-1, 60, 60, 1])
        
        #2 convolutional layers, 1 pooling layer
        h_conv1 = conv_layer(x_image, 1, 32)
        h_conv2 = conv_layer(h_conv1, 32, 64)
        
        h_pool1 = max_pool2d(h_conv2)
        
        #2 convolutional layers, 1 pooling layer
        h_conv3 = conv_layer(h_pool1, 64, 64)
        h_conv4 = conv_layer(h_conv3, 64, 64)
        
        h_pool2 = max_pool2d(h_conv4)
        
        # Fully connected layer
        fc1_weight = weight_variable([15*15*64, 2048])
        fc1_bias   = weight_variable([2048])
        
        # Reshape pooling output to fit fully connected layer input
        h_pool3_flat = tf.reshape(h_pool2, [-1, fc1_weight.get_shape().as_list()[0]])
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, fc1_weight), fc1_bias))
        # Apply Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)
        
        fc2_weight = weight_variable([2048, self.n_classes])
        fc2_bias   = weight_variable([self.n_classes])
        
        # Output, class prediction
        out = tf.add(tf.matmul(h_fc1_drop, fc2_weight), fc2_bias)
        return out
#Wrappers
def conv2d(x, W, b):
    #Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool2d(x):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, input_size, output_size):
    conv_weight = weight_variable([5, 5, input_size, output_size])
    conv_bias = bias_variable([output_size])

    h_conv = conv2d(input, conv_weight, conv_bias)
    return h_conv
    
