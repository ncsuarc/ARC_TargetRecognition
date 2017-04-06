import tensorflow as tf
import os
import traceback

from tfsession import TFSession

class Model:
    def __init__(self, model_path, batch_size = 100, display_step = 5):
        self.model_path = model_path
        self.batch_size = batch_size
        self.display_step = display_step
        self.sess = TFSession().sess

        try:
            path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.model_path
            ckpt = tf.train.get_checkpoint_state(path)
            print("Reading saved model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except Exception as e:
            raise ValueError("Error loading model: %s" % traceback.format_exc())

        self.input = tf.get_collection("image")[0]
        self.labels = tf.get_collection("labels")[0]
        self.keep_prob = tf.get_collection("kp")[0]
        self.predictor = tf.get_collection("predictor")[0]
        self.global_step = tf.get_collection("step")[0]
    
    def train(self, images, labels, n_classes, learning_rate = 0.001, dropout = 0.75):
        training_ops = tf.get_collection("training_ops")
        if len(training_ops) == 0:
            labels_one_hot = tf.one_hot(self.labels, n_classes)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictor, labels=labels_one_hot))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=self.global_step)
            correct_pred = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(labels_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.add_to_collection("training_ops", labels_one_hot) 
            tf.add_to_collection("training_ops", cost) 
            tf.add_to_collection("training_ops", optimizer) 
            tf.add_to_collection("training_ops", correct_pred) 
            tf.add_to_collection("training_ops", accuracy) 
        else:
            labels_one_hot = training_ops[0]
            cost = training_ops[1]
            optimizer = training_ops[2]
            correct_pred = training_ops[3]
            accuracy = training_ops[4]

        self.sess.run(tf.global_variables_initializer())

        step = 1
        while step * self.batch_size < len(images):
            batch_x = images[(step-1)*self.batch_size:step*self.batch_size]
            batch_y = labels[(step-1)*self.batch_size:step*self.batch_size]
            # Run optimization
            self.sess.run(optimizer, feed_dict={self.input: batch_x, self.labels: batch_y, self.keep_prob: dropout})
            if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc= self.sess.run([cost, accuracy], feed_dict={self.input: batch_x, self.labels: batch_y, self.keep_prob: 1.})

                    # Save the variables to disk.
                    save_path = self.saver.save(self.sess, self.model_path + '/model', global_step=self.global_step)

                    print("Checkpoint saved in file: %s" % save_path)
                    print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            step += 1
        save_path = self.saver.save(self.sess, self.model_path + '/model', global_step=self.global_step)
        print("Final checkpoint saved in file: %s" % save_path)

    def test(self, images):
        labels = []
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size]
            labels.extend(self.sess.run(self.predictor, feed_dict={self.input: batch_x, self.keep_prob: 1.}))
            step += 1
        return labels 

