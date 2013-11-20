import csv
import json
import numpy as np
import itertools
import mlpy
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

__all__ = ['FreshnessModel', 'PerceptronModel', 'RandomForestModel']

def index_or_none(l, v):
  try:
    return l.index(v)
  except ValueError:
    return None

def type_or_none(t, v):
  try:
    return t(v)
  except ValueError:
    return np.nan

class FreshnessModel(object):
  def __init__(self, trainfile, testfile, extra_features_file = None):
    self.clf = None
    self.data, self.target = self.default_features(trainfile, index_col = "urlid", secondary = {'label': int})
    self.target = {idx : label[0] for idx, label in self.target.iteritems()}
    self.test_data, _ = self.default_features(testfile, index_col = "urlid", secondary = {'urlid': int})
    self.test_ids = self.test_data.keys()
    if extra_features_file:
      with open(extra_features_file) as f:
        extra_features = json.load(f)
        for k, v in extra_features.iteritems():
          if k in self.data:
            self.data[k].extend(v)
          if k in self.test_data:
            self.data[k].extend(v)
    self.preprocess()

  def default_features(self, filename, index_col, secondary = []):
    features = {
      'avglinksize' : float,
      'commonLinkRatio_1' : float,
      'commonLinkRatio_2' : float,
      'commonLinkRatio_3' : float,
      'commonLinkRatio_4' : float,
      'compression_ratio' : float,
      'embed_ratio' : float,
      'frameBased' : int,
      'frameTagRatio' :  float,
      'hasDomainLink' : int,
      'html_ratio' : float,
      'image_ratio' : float,
      'is_news' : int,
      'lengthyLinkDomain' : int,
      'linkwordscore' : int,
      'news_front_page' : int,
      'non_markup_alphanum_characters' : int,
      'numberOfLinks' : int,
      'numwords_in_url' : float,
      'parametrizedLinkRatio' : float,
      'spelling_errors_ratio' : float,
    }
    with open(filename, 'rb') as f:
      reader = csv.reader(f, delimiter = '\t')
      headers = reader.next()
      id_idx = index_or_none(headers, index_col)
      indices = {index_or_none(headers, name):tp for name, tp in features.iteritems() if index_or_none(headers, name) is not None}
      secondary_indices = {index_or_none(headers, name):tp for name, tp in secondary.iteritems() if index_or_none(headers, name) is not None}
      # import ipdb; ipdb.set_trace()
      rows = {}
      secondary_rows = {}
      for row in reader:
        rows[int(row[id_idx])] = [type_or_none(t, row[i]) for i, t in indices.iteritems()]
        secondary_rows[int(row[id_idx])] = [type_or_none(t, row[i]) for i, t in secondary_indices.iteritems()]

      return rows, secondary_rows

  def preprocess(self):
    # impute missing values
    true_ids = set([urlid for urlid, label in self.target.iteritems() if label])
    true_data = [v for k, v in self.data.iteritems() if k in true_ids]
    false_data = [v for k, v in self.data.iteritems() if k not in true_ids]
    self.target = [1 for x in xrange(len(true_data))] + [0 for x in xrange(len(false_data))]
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    true_data = imp.fit_transform(true_data)
    false_data = imp.fit_transform(false_data)
    self.data = np.concatenate((true_data, false_data), axis=0)
    self.test_data = imp.fit_transform(self.test_data.values())

  def train(self, **kw):
    pass
  
  def pred(self, **kw):
    pass

  def score(self, data, target = None):
    if target is None:
      target = self.target
    predicted = self.pred(data)
    recall = metrics.recall_score(target, predicted, average = 'weighted')
    precision = metrics.precision_score(target, predicted, average = 'weighted')
    f1 = metrics.f1_score(target, predicted, average = 'weighted')
    return precision, recall, f1

  def test_output(self, outfile = 'submission.csv'):
    with open(outfile, 'wb') as f:
      writer = csv.writer(f)
      writer.writerow(['urlid','label'])
      writer.writerows(itertools.izip(map(int, self.test_ids), self.pred(self.test_data)))

  def cross_validation(self, num_splits = 4):
    scores = {'f1': [], 'precision':[], 'recall' : []}
    for i in xrange(num_splits):
      X_train, X_test, y_train, y_test = cross_validation.train_test_split(self.data, self.target, test_size = 0.4)
      self.train(X_train, y_train)
      p, r, f = self.score(X_test, y_test)
      scores['precision'].append(p)
      scores['f1'].append(f)
      scores['recall'].append(r)
    scores = {k: (np.array(v).mean(), np.array(v).std() * 2) for k, v in scores.iteritems()}
    return scores

class PerceptronModel(FreshnessModel):

  def __init__(self, trainfile, testfile, alpha = 0.1, thr = 0.05, maxiters = 1000):
    super(PerceptronModel, self).__init__(trainfile, testfile)
    self.clf = mlpy.Perceptron(alpha, thr, maxiters)
    self.train(data = self.data, target = self.target)

  def train(self, data = None, target = None):
    if data is None:
      data = self.data
    if target is None:
      target = self.target
    self.clf.learn(data, target)

  def pred(self, X):
    return self.clf.pred(X)

class RandomForestModel(FreshnessModel):
  def __init__(self, trainfile, testfile):
    super(RandomForestModel, self).__init__(trainfile, testfile)
    self.clf = RandomForestClassifier(n_estimators=10, max_depth=None)

  def train(self, data = None, target = None):
    if data is None:
      data = self.data
    if target is None:
      target = self.target
    self.clf.fit_transform(data, target)

  def pred(self, X):
    return self.clf.predict(X)

if __name__ == '__main__':
  pass
