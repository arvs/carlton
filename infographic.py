import csv
import json

def write_to_files(csv_file):
  with open(csv_file, 'rb') as infile, open('boilerplate_text.txt', 'wb') as boil, open('titles.txt','wb') as titles: 
    reader = csv.reader(infile, delimiter = '\t')
    headers = reader.next()
    for line in reader:
      boiler = json.loads(line[2])
      titles.write("%s\n" % boiler['title'])
      boil.write("%s\n" % boiler['body'])

if __name__ == '__main__':
  write_to_files('train.tsv')
