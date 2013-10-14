import os, codecs, sys, random
from redis import StrictRedis
from random import sample

__all__ = ['r', 'FileCorpus']

r = StrictRedis(host='localhost', port=6379, db=0)

class Corpus(object):
  def __init__(self, corpus_name):
    self._name = corpus_name

  def docs(self, num_docs, train=True):
    raise NotImplementedError

class FileCorpus(Corpus):
  def __init__(self, dirname = None, extensions = None, name = None, ingest = True):
    name = name if name else dirname
    super(FileCorpus, self).__init__(corpus_name = name)
    self.name = name

    if ingest:
      self.num_docs = 0

      for filename in os.listdir(dirname):
        f_name, ext = os.path.splitext(os.path.join(dirname, filename))
        if ext in extensions:
          self.num_docs += 1
          with codecs.open("%s%s" % (f_name, ext), 'r', 'utf-8') as f:
            self[self.num_docs] = ["".join(f.readlines()), f_name]

      r.set("lda_%s_num_docs" % self.name, self.num_docs)
      r.set("lda_%s_slice_counter" % self.name, 1)
    else:
      self.num_docs = int(r.get("lda_%s_num_docs" % self.name))

    def key(self, num):
      return "%s_doc_%d" % (self.name, num)

    @classmethod
    def get(cls, name):
      return cls(name = name, ingest = False)

    def __getitem__(self, num):
      key = self.key(num)
      return r.lrange(key, 0, -1)

    def __setitem__(self, num, val):
      if not isinstance(val, list):
        print "Value Error: List expected", sys.exc_info()[0]
        raise
      else:
        key = self.key(num)
        r.delete(key)
        r.rpush(key, *val)

    def wipe_from_cache(self):
      for a in xrange(self.num_docs):
        key = self.key(a)
        r.delete(key)

    @property
    def _counter(self):
      return int(r.get("%s_slice_counter" % self.name))

    def _increment_slice(self,num):
      r.set("%s_slice_counter" % self.name, self._counter + num)

    def reset_counter(self):
      r.set("%s_slice_counter" % self.name, 1)

    def docs(self, num, deterministic = True):
      if deterministic:
        if self._counter + num >= self.num_docs:
          self.reset_counter()
        sliced = [self[n] for n in xrange(self._counter, self._counter + num)]
        self._increment_slice(num)
        return map(list, zip(*sliced))
      else:
        random_docs = [self[r] for r in random.sample(range(self.num_docs), num)]
        return map(list, zip(*random_docs))

