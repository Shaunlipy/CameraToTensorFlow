"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import pickle
import random
import numpy as np
import cv2


# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    return (x-x.min())/(x.max()-x.min())

def test_model():
    """
    Test the saved model against the test dataset
    """
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        small = cv2.resize(frame,(32,32))
        
        test_features = np.array([normalize(small)])
        test_labels = np.array([np.zeros(10,dtype=int)]) # no use just to hold places

        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
    #         # Load model
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

    #         # Get Tensors from loaded model
            loaded_x = loaded_graph.get_tensor_by_name('x:0')
            loaded_y = loaded_graph.get_tensor_by_name('y:0')
            loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            random_test_predictions = sess.run(
                tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
                feed_dict={loaded_x: test_features, loaded_y: test_labels, loaded_keep_prob: 1.0})
            
            ans = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print (ans[random_test_predictions.indices[0][0]],ans[random_test_predictions.indices[0][1]],ans[random_test_predictions.indices[0][2]])        
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            


test_model()