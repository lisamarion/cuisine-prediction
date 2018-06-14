import tensorflow as tf
import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
	print("Please specify a json file of recipes to classify.")
	sys.exit()

filename = sys.argv[1]

if filename[-5:] != '.json':
	print("Incorrect file format, please specify a json file.")
	sys.exit(2)

print("Reading data...")
raw_data_df = pd.read_json(filename)
ids_df = raw_data_df['id']
ingredients_df = raw_data_df.ingredients.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('ingredients')
ingredients_df['count'] = 1
features_df = ingredients_df.pivot_table(index=ingredients_df.index, columns='ingredients', values='count').fillna(0)
label_dict = pd.read_csv('trainY.csv', nrows=1, sep=',').columns

#match available features to training features
training_features = pd.read_csv('trainX.csv', nrows=1, sep=',').columns.values.tolist()
prediction_features = features_df.columns.values.tolist()
for feature in training_features:
	if feature not in prediction_features:
		features_df[feature] = 0
feature_difference = list(set(prediction_features) - set(training_features))
features_df.drop(feature_difference, axis=1, inplace=True)
features_df = features_df.reindex(sorted(features_df.columns), axis=1)
features = features_df.values
features = np.delete(features, 0, axis=1)

#load trained model
print("Loading trained model...")
graph = tf.get_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph('models/cuisine_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('models/'))
prediction = graph.get_tensor_by_name('output:0')
X = graph.get_tensor_by_name('X:0')
print("Done loading model.")

#iterate through data and make predictions
for i in range(0, features.shape[0]):
	recipe_id = ids_df[i]
	input_features = features[i]
	input_features = input_features[None, :]
	output_predictions = sess.run(prediction, feed_dict={X: input_features})
	top_three_indexes = []
	top_three_values = []
	for i in range(3):
		current_max_index = np.argmax(output_predictions)
		top_three_indexes.append(current_max_index) 
		top_three_values.append(output_predictions[0][current_max_index])
		output_predictions[0][current_max_index] = -1
	for index, value in enumerate(top_three_indexes):
		print(recipe_id, end=',')
		print(label_dict[value], end=',')
		print(top_three_values[index])