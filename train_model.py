import pandas as pd

def preprocess_data(data_path):
  # import data from CSV file
  df = pd.read_csv(data_path)

  # drop null values
  df.dropna(inplace=True)

  # split data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2)

  return X_train, X_test, y_train, y_test

from sklearn.mixture import GaussianMixture

def train_model(X_train, y_train):
  # create the Gaussian mixture model
  model = GaussianMixture(n_components=4)

  # fit the model to the training data
  model.fit(X_train, y_train)

  return model
