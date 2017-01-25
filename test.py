import tensorflow as tf
import cv2
import prepare_data
import model

images, labels = prepare_data.prep_data()



# Launch the graph
with tf.Session() as sess:
    cnn_model = model.Model(sess)
    pred_labels = cnn_model.test(sess, images)
    for (img, predicted, actual) in zip(images, pred_labels, labels):
        cv2.imshow("Display", img.reshape((cnn_model.img_height, cnn_model.img_width)))
        print("actual:%s predicted:%s" % (prepare_data.label_to_alphanumeric(actual), prepare_data.label_to_alphanumeric(predicted)))
        cv2.waitKey()
