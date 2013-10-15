import csv
import numpy as np
import mlpy
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Imputer

__all__ = ['FreshnessModel', 'PerceptronModel']

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
  def __init__(self, trainfile, testfile):
    self.clf = None
    self.data, self.target = self.default_features(trainfile, secondary = {'label': int})
    self.target = self.target.flatten()
    self.test_data, self.test_ids = self.default_features(testfile, secondary = {'urlid': int})
    self.test_ids = self.test_ids.flatten()

  def default_features(self, filename, secondary = []):
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
      indices = {index_or_none(headers, name):tp for name, tp in features.iteritems() if index_or_none(headers, name) is not None}
      secondary_indices = {index_or_none(headers, name):tp for name, tp in secondary.iteritems() if index_or_none(headers, name) is not None}
      # import ipdb; ipdb.set_trace()
      rows = []
      secondary_rows = []
      for row in reader:
        rows.append([type_or_none(t, row[i]) for i, t in indices.iteritems()])
        secondary_rows.append([type_or_none(t, row[i]) for i, t in secondary_indices.iteritems()])
      imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
      return imp.fit_transform(rows), imp.fit_transform(secondary_rows)

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
      writer.writerows(itertools.izip(self.test_ids, self.clf.pred(data = self.test_data)))

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

  def __init__(self, trainfile, testfile):
    super(PerceptronModel, self).__init__(trainfile, testfile)
    self.clf = mlpy.Perceptron(alpha=0.1, thr=0.05, maxiters=100)

  def train(self, data = None, target = None):
    if data is None:
      data = self.data
    if target is None:
      target = self.target
    self.clf.learn(data, target)

  def pred(self, X):
    return self.clf.pred(X)

if __name__ == '__main__':
  pass
