import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import neighbors
from sklearn import utils


_CHANNELS = ['channel_{}'.format(i) for i in range(8)]

def read_clean(filename, pos_label, neg_label):
  """Reads a datafile and drops samples without desired labels."""
  raw_data = pd.read_csv(filename, sep=',') # returns a 2d array
  pos_idx = raw_data['tag'] == pos_label # returns an array of true or false if matches pos_label
  neg_idx = raw_data['tag'] == neg_label # returns an array of true or false if matches neg_label
  return raw_data[pos_idx | neg_idx] # returns an array sans anything that isn't either pos or neg_label

def prepare_train_test(data):
  """Prepares data for training. Returns Xtr, Ytr, Xte, Yte."""
  shuffled = utils.shuffle(data)
  X = shuffled[_CHANNELS]
  y = shuffled['tag']
  return cross_validation.train_test_split(X, y, test_size=0.2)

class TransformClassifier(object):
  def __init__(self, transformer, classifier):
    self.tr = transformer
    self.clf = classifier

  def fit(self, X, y):
    X_transformed = self.tr.fit_transform(X)
    self.clf.fit(X_transformed, y)

  def predict(self, X):
    X_transformed = self.tr.transform(X)
    return self.clf.predict(X_transformed)

  def score(self, X, y):
    X_transformed = self.tr.transform(X)
    return self.clf.score(X_transformed, y)

if __name__ == '__main__':
  assert len(sys.argv) == 2, "Must provide filename as argument."

  # Read in file.
  data = read_clean(sys.argv[1], 'go', 'stop')

  # Split into train/test
  Xtr, Xte, Ytr, Yte = prepare_train_test(data)

  # Prepare submodels
  ica = decomposition.FastICA(n_components=6)
  nn = neighbors.KNeighborsClassifier(n_neighbors=3)
  model = TransformClassifier(ica, nn)

  # Train model
  model.fit(Xtr, Ytr)
  print(model.score(Xte, Yte))
