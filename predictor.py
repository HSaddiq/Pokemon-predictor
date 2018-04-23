import pandas as pd
import tensorflow as tf
import numpy as np

'''x_test = np.array([[1,100,110,90,85,90,60,4,95,109,105,75,85,56]])
x_test = pd.DataFrame(x_test)
x_test = x_test.as_matrix()
print(x_test)

feed_dict = {
    x: x_test
    }
print(sess.run(y, feed_dict))'''


def predict(pokemon_name_1, pokemon_name_2):
    #initialising variables

    input_nodes = 14

    hidden_nodes1 = 10
    hidden_nodes2 = 20

    pkeep = tf.placeholder(tf.float32)

    # input
    x = tf.placeholder(tf.float32, [None, input_nodes])

    # layer 1
    W1 = tf.Variable(
        tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_nodes1]))
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # layer 2
    W2 = tf.Variable(
        tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_nodes2]))
    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

    # layer 3
    W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, 1], stddev=0.15))
    b3 = tf.Variable(tf.zeros([1]))
    y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

    # output
    y = y3
    y_ = tf.placeholder(tf.float32, [None, 1])

    # initialising tensorflow
    with tf.Session() as sess:
        # import previously generated model
        saver = tf.train.import_meta_graph(
            "./trained_model/trained_predictor.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./trained_model/."))

        #initialise variables and graph
        sess.run(tf.global_variables_initializer())

        graph = tf.get_default_graph()

        x_test = np.array(
            [[1, 100, 110, 90, 85, 90, 60, 4, 95, 109, 105, 75, 85, 56]])
        x_test = pd.DataFrame(x_test)
        x_test = x_test.as_matrix()

        feed_dict = {
            x: x_test
        }

        prediction = sess.run(y, feed_dict)

        return prediction

print(predict(1,1))
