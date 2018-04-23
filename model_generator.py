import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv("data/cleaned_data.csv")

pokemon_1_win_dataframe = df[df.Outcome == 0]
pokemon_2_win_dataframe = df[df.Outcome == 1]

# make training dataset with 80% of data
X_train = pd.concat([pokemon_1_win_dataframe.sample(frac=0.8),
                     pokemon_2_win_dataframe.sample(frac=0.8)], axis=0)

# make test dataset with remaining data
X_test = df.loc[~df.index.isin(X_train.index)]

# shuffle data to ensure appropriate learning process
X_train = X_train.sample(frac=1)
X_test = X_test.sample(frac=1)

y_train = X_train.Outcome
y_test = X_test.Outcome

# remove outcome from X_train and X_test
X_train = X_train.drop(['Outcome'], axis=1)
X_test = X_test.drop(['Outcome'], axis=1)

# Split the testing data in half
split = int(len(y_test) / 2)

inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]

inputY = inputY[:, np.newaxis]
inputY_valid = inputY_valid[:, np.newaxis]
inputY_test = inputY_test[:, np.newaxis]

print(inputY)

input_nodes = 14

hidden_nodes1 = 10
hidden_nodes2 = 20

pkeep = tf.placeholder(tf.float32)

# input
x = tf.placeholder(tf.float32, [None, input_nodes])

# layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15))
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

# Parameters
training_epochs = 4
training_dropout = 0.9
display_step = 1
n_samples = y_train.shape[0]
batch_size = 20
learning_rate = 0.005

# Cost function: root mean square error
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y))))

# We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction if the most likely y value from softmax equals the target value.


correct_prediction = tf.equal(tf.round(y), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy_summary = []  # Record accuracy values for plot
cost_summary = []  # Record cost values for plot
valid_accuracy_summary = []
valid_cost_summary = []
stop_early = 0  # To keep track of the number of epochs before early stopping

# Save the best weights so that they can be used to make the final predictions
# checkpoint = "location_on_your_computer/best_model.ckpt"
saver = tf.train.Saver(max_to_keep=1)

# Initialize variables and tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples / batch_size)):
            batch_x = inputX[batch * batch_size: (1 + batch) * batch_size]
            batch_y = inputY[batch * batch_size: (1 + batch) * batch_size]

            sess.run([optimizer], feed_dict={x: batch_x,
                                             y_: batch_y,
                                             pkeep: training_dropout})

        # Display logs after every epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost],
                                               feed_dict={x: inputX,
                                                          y_: inputY,
                                                          pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost],
                                                     feed_dict={x: inputX_valid,
                                                                y_: inputY_valid,
                                                                pkeep: 1})

            print("Epoch:", epoch,
                  "Acc =", "{:.10f}".format(train_accuracy),
                  "Cost =", "{:.10f}".format(newCost),
                  "Valid_Acc =", "{:.10f}".format(valid_accuracy),
                  "Valid_Cost = ", "{:.10f}".format(valid_newCost))

            # Save the weights if these conditions are met.
            # if epoch > 0 and valid_accuracy > max(valid_accuracy_summary) and valid_accuracy > 0.999:
            #    saver.save(sess, checkpoint)

            # Record the results of the model
            accuracy_summary.append(train_accuracy)
            cost_summary.append(newCost)
            valid_accuracy_summary.append(valid_accuracy)
            valid_cost_summary.append(valid_newCost)

            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0

    # Saves the trained model for use in predictor.py
    saver.save(sess, "./trained_model/trained_predictor")

    # the following is a test to see the functionality of the model

    '''x_test = np.array([[1,100,110,90,85,90,60,4,95,109,105,75,85,56]])
    x_test = pd.DataFrame(x_test)
    x_test = x_test.as_matrix()
    print(x_test)

    feed_dict = {
        x: x_test
        }
    print(sess.run(y, feed_dict))'''

    print()
    print("Optimization Finished!")
    print()
