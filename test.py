import tensorflow as tf
import cv2

import prepare_data
import model

#Hyper params
learning_rate = 0.001
batch_size = 100
display_step = 1000

#Graph params
n_input = 3600 # 60x60 Images
n_classes = 13 # Output Classes 
dropout = 0.75 # Keep Probability

# Construct model
cnn_model = model.Model(n_input, n_classes, dropout)
pred = cnn_model.predictor
x = cnn_model.x
y = cnn_model.y
keep_prob = cnn_model.keep_prob

# Initializing the variables
init = tf.initialize_all_variables()

#For saving trained Neural Net for later
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
        with tf.device('/cpu:0'):
                sess.run(init)
                # Restore variables from disk.
                ckpt = tf.train.get_checkpoint_state('./training')
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                images, labels = prepare_data.prep_data()
                step = 0
                while step * batch_size < len(images):
                        batch_x = images[step*batch_size:(step+1)*batch_size]
                        batch_y = labels[step*batch_size:(step+1)*batch_size]
                        # Run optimization op (backprop)
                        pred_labels = sess.run(pred, feed_dict={x: batch_x})
                        for (img,predicted,actual) in zip(batch_x, pred_labels, batch_y):
                                cv2.imshow("Display", img.reshape((60,60)))
                                print("%s : %s ::: %s" % (prepare_data.label_to_int(predicted), actual, prepare_data.label_to_shape(predicted)))
                                cv2.waitKey()
                        step += 1
