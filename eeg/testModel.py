import sys

import pandas as pd
from sklearn import linear_model
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


def prepare_train_test(train_data, test_data):
    """Prepares data for training. Returns Xtr, Ytr, Xte, Yte."""
    shuffled_training = utils.shuffle(train_data)
    shuffled_test = utils.shuffle(test_data)
    Xtr = shuffled_training[_CHANNELS]
    ytr = shuffled_training['tag']
    Xte = shuffled_test[_CHANNELS]
    yte = shuffled_test['tag']
    return Xtr, Xte, ytr, yte


class TransformClassifier(object):
    def __init__(self, transformer, classifier):
        self.tr = transformer
        self.clf = classifier

    def fit(self, X, y):
        # X_transformed = self.tr.fit_transform(X)
        self.clf.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.tr.transform(X)
        return self.clf.predict(X_transformed)

    def score(self, X, y):
        X_transformed = self.tr.transform(X)
        return self.clf.score(X_transformed, y)


if __name__ == '__main__':
    assert len(sys.argv) == 3, "Must provide 2 filenames as arguments."

    # Read in file.
    train_data = read_clean(sys.argv[1], 'go', 'stop')
    test_data = read_clean(sys.argv[2], 'go', 'stop')

    # Split into train/test
    Xtr, Xte, Ytr, Yte = prepare_train_test(train_data, test_data)

    # Prepare submodels
    # ica = decomposition.FastICA(n_components=6)
    # nn = neighbors.KNeighborsClassifier(n_neighbors=5)

    model = linear_model.LogisticRegression(C=1e5)

    # Train model
    model.fit(Xtr, Ytr)
    print(model.score(Xte, Yte))
