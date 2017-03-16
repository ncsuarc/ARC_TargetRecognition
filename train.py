import tensorflow as tf
import prepare_data
import model

images, labels = prepare_data.prep_data()

# Launch the graph
cnn_model = model.Model('training', batch_size=500)
cnn_model.train(images, labels)
