import pandas as pd
import numpy as np

def process_recipe_data(input_file, output_file):
	#read raw data into dataframe
	raw_data_df = pd.read_json(input_file)

	#generate binary-encoded ingredient feature columns
	ingredients_df = raw_data_df.ingredients.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('ingredients')
	ingredients_df['count'] = 1
	features_df = ingredients_df.pivot_table(index=ingredients_df.index, columns='ingredients', values='count').fillna(0)
	
	#remove features that occur in fewer than 10 recipes
	features_df.drop([feature for feature, total in features_df.sum().iteritems() if total < 10], axis=1, inplace=True)

	#generate binary-encoded label columns
	cuisine_df = raw_data_df.cuisine.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('cuisine')
	cuisine_df['count'] = 1
	labels_df = cuisine_df.pivot_table(index=cuisine_df.index, columns='cuisine', values='count').fillna(0)

	#strip commas from column names and write to csv
	features_df.columns = [x.strip().replace(',', '') for x in features_df.columns]
	features_df.to_csv(output_file + 'X.csv')
	labels_df.to_csv(output_file + 'Y.csv')

print("Starting train data...")
process_recipe_data('cuisine.train.json', 'train')
print("Starting test data...")
process_recipe_data('cuisine.dev.json', 'test')