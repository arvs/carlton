from __future__ import division
import json
import csv
import re
import calendar
import numpy as np
from collections import Counter
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

class FeatureGenerator(object):

  def __init__(self, trainfile, testfile):
    with open(trainfile) as train, open(testfile) as test:
      self.features = self.nlp_features(csv.reader(train, delimiter='\t'))
      self.features.update(self.nlp_features(csv.reader(train, delimiter='\t')))

  def nlp_features(self, row_iter):
    count = 0
    headers = row_iter.next()
    boiler_idx = headers.index('boilerplate')
    urlid_idx = headers.index('urlid')
    features = {}
    divide_if_none = lambda a,b: a/b if b != 0 else 0
    months = [calendar.month_name[i].lower() for i in range(1, 12)]
    months.extend([calendar.month_abbr[i].lower() for i in range(1, 12)])
    for row in row_iter:
      count += 1
      body = word_tokenize(row[boiler_idx])
      lowered_body = word_tokenize(row[boiler_idx].lower())
      tags = pos_tag(body)
      tag_counts = Counter((t[1] for t in tags))
      agg_tags = lambda tags: sum([tag_counts.get(t, 0) for t in tags])
      body_len = len(body)
      num_unique_words = len(set(lowered_body))
      num_cap_words = len(filter(lambda v: re.match("[A-Z].*", v), body))
      avg_words_per_sentence = divide_if_none(body_len, num_cap_words)
      num_mth_words = len(filter(lambda v: v in months, lowered_body))
      num_nums = tag_counts.get('CD', 0)
      num_to_word_ratio = divide_if_none(num_nums, body_len)
      num_foreign = tag_counts.get('FW', 0)
      avg_word_len = np.mean(map(len, body))
      agg_tags = lambda tags: sum([tag_counts.get(t, 0) for t in tags])
      num_nouns = agg_tags(['NN', 'NNP', 'NNPS', 'NNS'])
      noun_to_word_ratio = divide_if_none(num_nouns, body_len)
      proper_noun_ratio = divide_if_none(agg_tags(['NNP', 'NNPS']), num_nouns)
      num_conj = tag_counts.get('CC', 0)
      conj_ratio = divide_if_none(num_conj, body_len)
      len_words_no_sym_det = sum([v for k,v in tag_counts.iteritems() if k not in ('DT', '$', '(', ')', ',', '--', '.', ':', 'POS', 'SYM')])
      num_verbs = agg_tags(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
      verb_ratio = divide_if_none(num_verbs, body_len)
      past_to_pres_tense_ratio = divide_if_none(agg_tags(['VBN', 'VBD']), num_verbs)
      num_adj = agg_tags(['JJ', 'JJR', 'JJS'])
      num_list_markers = tag_counts.get('LS', 0)

      features[int(row[urlid_idx])] = [body_len, num_unique_words, avg_words_per_sentence, num_cap_words, num_mth_words, num_nums, num_to_word_ratio, num_foreign, avg_word_len, num_nouns, noun_to_word_ratio, proper_noun_ratio, num_conj, conj_ratio, len_words_no_sym_det, num_verbs, verb_ratio, past_to_pres_tense_ratio, num_adj, num_list_markers]

      if count % 100 == 0:
        print count

  def write_to_file(self, filename):
    with open(filename, 'wb') as f:
      json.dump(self.features, f)

if __name__ == '__main__':
  fg = FeatureGenerator('train.tsv','test.tsv')
  fg.write_to_file('extra_features_kaggle.json')