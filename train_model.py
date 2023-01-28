import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import linalg

def train(data):
  # create the Gaussian mixture model
  model = mixture.BayesianGaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(data)

  # We would want to look at model.predict(data), model.means_, model.covariances_
  return model

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    plt.ylim(-5.0, 5.0)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    
if __name__ == '__main__':
  Xs = np.random.uniform(0, 100, (100, 2))
  Ys = np.random.randint(0, 100, (5, 2))
  test_model = train(Xs)
  print(test_model.predict(Ys))
  
  # def preprocess_data(data_path):
#   # import data from CSV file
#   df = pd.read_csv(data_path)

#   # drop null values
#   df.dropna(inplace=True)

#   # split data into training and test sets
#   X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2)

#   return X_train, X_test, y_train, y_test

# def main():
#     # preprocess data
#     X_train, X_test, y_train, y_test = preprocess_data('data.csv')

#     # train model
#     model = train(X_train, y_train)

#     # identify market condition
#     probabilities = identify_market_condition(X_test)

#     # save model
#     joblib.dump(model, 'model.joblib')