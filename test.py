import tensorflow as tf
import cv2
import prepare_data
import model

images, labels = prepare_data.prep_data("test_samples")

# Launch the graph
cnn_model = model.Model('training', batch_size=500)
pred_labels = cnn_model.test(images)

correct = 0
total = 0
n = 0

for (img, predicted, actual) in zip(images, pred_labels, labels):
    cv2.imshow("Display", img.reshape((60, 60, 3)))
    actual = prepare_data.label_to_shape(actual)
    predicted = prepare_data.label_to_shape(predicted)
    total = total + 1
    if actual == predicted:
        correct = correct + 1
    else:
        cv2.imwrite("bad%s.png" % n, img.reshape((60, 60, 3)))
        n += 1
    print("actual:%s predicted:%s" % (actual, predicted))
    print('Percent Correct:%6.2f%%' % (correct/total * 100))
    cv2.waitKey()
