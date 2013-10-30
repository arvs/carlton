import csv
import json
import codecs

def write_to_files(csv_file):
  with codecs.open(csv_file, 'rb') as infile, codecs.open('boilerplate_text.txt', 'w','utf-8') as boil, codecs.open('titles.txt','w','utf-8') as titles: 
    reader = csv.reader(infile, delimiter = '\t')
    headers = reader.next()
    for line in reader:
      boiler = json.loads(line[2])
      # import ipdb; ipdb.set_trace()
      titles.write("%s\n" % unicode(boiler.get('title','')))
      boil.write("%s\n" % unicode(boiler.get('body','')))

if __name__ == '__main__':
  write_to_files('train.tsv')