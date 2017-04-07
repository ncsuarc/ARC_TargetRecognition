import tensorflow as tf
import json
import copy
import os

class JSONObject:
    def __init__(self, json):
        for attr, val in json.items():
            if isinstance(val, (list, tuple)):
               setattr(self, attr, [JSONObject(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, attr, JSONObject(val) if isinstance(val, dict) else val)

def create_network(save_location, json_location): 
    with open(json_location, 'r') as data_file:
        architecture = JSONObject(json.load(data_file))

    sess = tf.Session()

    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, architecture.img_height * architecture.img_width * architecture.color_channels], name='image')
        tf.add_to_collection("image", image)

        global_step = tf.Variable(0, name='global_step', trainable=False) 
        tf.add_to_collection("step", global_step)
    
    predictor = conv_net(image, architecture)
    softmax = tf.nn.softmax(predictor)
    tf.add_to_collection("predictor", predictor)
    tf.add_to_collection("predictor", softmax)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, save_location, global_step=global_step)
    print("Saved initial model in file: %s" % save_path)

def conv_net(x, architecture):
    img_height = architecture.img_height
    img_width = architecture.img_width
    color_channels = architecture.color_channels
    
    with tf.name_scope('input'):
        keep_prob = tf.placeholder(tf.float32) 
        tf.add_to_collection("kp", keep_prob)
    #Reshape input image
    x_image = tf.reshape(x, shape=[-1, img_height, img_width, color_channels])
    
    prev_s = copy.deepcopy(architecture.layers[0])
    prev_s.shape = color_channels
    prev_s.type = ''
    layer = x_image
    for layer_s, i in zip(architecture.layers, range(len(architecture.layers))):
        with tf.name_scope(layer_s.type + str(i)):
            if layer_s.type == 'conv':
                layer = conv_layer(layer, prev_s.shape, layer_s.shape)
                prev_s = layer_s
                print('Convolving %d features to %d features.' % (prev_s.shape, layer_s.shape))
            elif layer_s.type == 'max_pool':
                if (img_height % layer_s.shape != 0) or (img_width % layer_s.shape != 0):
                    raise ValueError('Invalid dimensions as input to pooling layer')

                layer = max_pool2d(layer, k=layer_s.shape)
                print('Pooling from %d x %d to %d x %d' % (img_height, img_width, img_height / layer_s.shape, img_width / layer_s.shape))
                img_height = img_height / layer_s.shape
                img_width = img_width / layer_s.shape
            elif layer_s.type == 'fc':
                if prev_s.type != 'fc':
                    prev_s.shape *= int(img_height * img_width)
                    layer = tf.reshape(layer, [-1, prev_s.shape])
                print('Fully connected layer from %d neurons to %d neurons.' % (prev_s.shape, layer_s.shape))
                weight = weight_variable([prev_s.shape, layer_s.shape])
                bias   = bias_variable([layer_s.shape])
                
                # Reshape pooling output to fit fully connected layer input
                layer = tf.add(tf.matmul(layer, weight), bias)
                if layer_s.relu:
                    layer = tf.nn.relu(layer)
                
                if layer_s.dropout:
                    layer = tf.nn.dropout(layer, keep_prob)

                prev_s = layer_s
            else:
                raise ValueError("Unknown layer type: {}".format(layer.type))

    return layer

#Wrappers
def conv2d(x, W, b):
    #Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
    x = tf.nn.bias_add(x, b, name='bias_add')
    return tf.nn.relu(x, name='relu')

def max_pool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name='pool')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def conv_layer(layer_in, input_size, output_size):
    conv_weight = weight_variable([5, 5, input_size, output_size])
    conv_bias = bias_variable([output_size])

    return conv2d(layer_in, conv_weight, conv_bias)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a model')
    parser.add_argument("-s", "--save-location", required=True, help="Directory to save training in.")
    parser.add_argument("-a", "--architecture", required=True, help="JSON file containing the architecture information about the model.")

    args = parser.parse_args()
    create_network(os.path.join(args.save_location, "model"), args.architecture) 
