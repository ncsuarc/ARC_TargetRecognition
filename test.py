import tensorflow as tf
import cv2
import prepare_data
import model

images, labels = prepare_data.prep_data()



# Launch the graph
with tf.Session() as sess:
    cnn_model = model.Model(sess)
    pred_labels = cnn_model.test(sess, images)
    correct = 0
    total = 0
    for (img, predicted, actual) in zip(images, pred_labels, labels):
        cv2.imshow("Display", img.reshape((cnn_model.img_height, cnn_model.img_width, cnn_model.color_channels)))
        actual = prepare_data.label_to_shape(actual)
        predicted = prepare_data.label_to_shape(predicted)
        total = total + 1
        if actual == predicted:
            correct = correct + 1
        print("actual:%s predicted:%s" % (actual, predicted))
        print('Percent Correct:%6.2f%%' % (correct/total * 100))
        cv2.waitKey()
