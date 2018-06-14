import tensorflow as tf
import pandas as pd
import numpy as np

def import_data_from_csv():
	trainX_df = pd.read_csv('trainX.csv', sep=',')
	trainY_df = pd.read_csv('trainY.csv', sep=',')
	testX_df = pd.read_csv('testX.csv', sep=',')
	testY_df = pd.read_csv('testY.csv', sep=',')

	#match validation features to training features
	for ingredient in trainX_df.columns:
		if ingredient not in testX_df.columns:
			testX_df[ingredient] = 0
	for ingredient in testX_df.columns:
		if ingredient not in trainX_df.columns:
			testX_df.drop(ingredient, axis=1, inplace=True)
	testX_df = testX_df.reindex(sorted(testX_df.columns), axis=1)

	trainX = trainX_df.values
	trainX = np.delete(trainX, 0, axis=1)
	trainY = trainY_df.values
	trainY = np.delete(trainY, 0, axis=1)
	testX = testX_df.values
	testX = np.delete(testX, 0, axis=1)
	testY = testY_df.values
	testY = np.delete(testY, 0, axis=1)
	return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data_from_csv()

#dataset parameters
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]
print("Number of features: " + str(numFeatures))
print("Number of labels: " + str(numLabels))

#training parameters
numEpochs = 10000
learningRate = 0.08

#placeholders
X = tf.placeholder(tf.float32, [None, numFeatures], name='X')
Y = tf.placeholder(tf.float32, [None, numLabels])

#variables
weights = tf.Variable(tf.random_normal([numFeatures, numLabels], mean=0, stddev=(np.sqrt(6/(numFeatures+numLabels+1))), name="weights"))
bias = tf.Variable(tf.random_normal([1, numLabels], mean=0, stddev=(np.sqrt(6/(numFeatures+numLabels+1))), name="bias"))

#operations
initialize = tf.global_variables_initializer()
apply_weights = tf.matmul(X, weights)
add_bias = tf.add(apply_weights, bias)
output = tf.nn.softmax(add_bias, name='output')
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=add_bias))
optimization = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_op)

#run training session
sess = tf.Session()
sess.run(initialize)
correctPredictions = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

cost = 0
diff = 1

for i in range(numEpochs):
	if i > 1 and diff < .0001:
		print("Change in cost %g; reached convergence." % diff)
		break
	else:
		step = sess.run(optimization, feed_dict={X: trainX, Y: trainY})
		#show progress every 10 epochs
		if i % 10 == 0:
			trainAccuracy, newCost = sess.run([accuracy, cost_op], feed_dict={X: trainX, Y: trainY})
			diff = abs(newCost - cost)
			cost = newCost
			print("Step %d, training accuracy %g" % (i, trainAccuracy))
			print("Step %d, cost %g" % (i, newCost))
			print("Step %d, change in cost %g" % (i, diff))

print("Final accuracy on test set: %g" % sess.run(accuracy, feed_dict={X: testX, Y: testY}))

#save model variables
saver = tf.train.Saver()
saver.save(sess, 'models/cuisine_model')
sess.close()