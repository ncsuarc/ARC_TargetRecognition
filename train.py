import tensorflow as tf

import prepare_data
import model

import random

#Hyper params
learning_rate = 0.001
batch_size = 100
display_step = 5

#Graph params
n_input = 3600 # 60x60 Images
n_classes = 36 # Output Classes
dropout = 0.75 # Keep Probability

# Construct model
cnn_model = model.Model(n_input, n_classes, dropout)
pred = cnn_model.predictor
x = cnn_model.x
y = cnn_model.y
y_one_hot = cnn_model.y_one_hot
keep_prob = cnn_model.keep_prob
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#For saving trained Neural Net for later
saver = tf.train.Saver(tf.trainable_variables())
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	# Keep training until reach max iterations
	images, labels = prepare_data.prep_data()
	step = 0
	while step * batch_size < len(images):
	        batch_x = images[step*batch_size:(step+1)*batch_size] #Take one image from every character
	        batch_y = labels[step*batch_size:(step+1)*batch_size]
	        # Run optimization op (backprop)
	        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
	        if step % display_step == 0:
	                # Calculate batch loss and accuracy
	                loss, acc, pred_y = sess.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

	                # Save the variables to disk.
	                save_path = saver.save(sess, "training/model", global_step=step)

	                print("Checkpoint saved in file: %s" % save_path)
	                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
	                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                          "{:.5f}".format(acc))
	        step += 1
	print("Optimization Finished!")


	# Calculate accuracy for 2 random batches
	random.seed()
	step = random.randint(0, 508)

	print("Testing Accuracy:", \
	        sess.run(accuracy, feed_dict={x: images[step::508],
	                                                                  y: labels[step::508],
	                                                                  keep_prob: 1.}))
