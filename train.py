import tensorflow as tf
import prepare_data
import model

images, labels = prepare_data.prep_data()

# Launch the graph
with tf.Session() as sess:
    cnn_model = model.Model(sess, False)
    cnn_model.train(sess, images, labels)
