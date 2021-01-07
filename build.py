import json
import argparse
import collections
from Tree import getProductions
import os

def processNlToks(nlToks):
  return [tok.encode('ascii', 'replace').decode().strip() for tok in nlToks \
           if tok != "-RCB-" and \
           tok != "-LCB-" and \
           tok != "-LSB-" and \
           tok != "-RSB-" and \
           tok != "-LRB-" and \
           tok != "-RRB-" and \
           tok != "@link" and \
           tok != "@code" and \
           tok != "@inheritDoc" and \
           tok.encode('ascii', 'replace').decode().strip() != '']

trainNls = []

def processFiles(fname, prefix, dset):
  dataset = []

  i = 0
  import pickle
  with open(fname, 'rb') as file:
    data = pickle.load(file)
  for d in data:
    code = d['query']
    nl = d['question']
    rule_seq = d['action_rules']

    seq2seq = nl.split()


    dataset.append(
      {'nl': nl,
       'code': code,
       'rules': rule_seq,
       'seq2seq': seq2seq,
       }
    )

  f = open(prefix + '.dataset', 'w')
  f.write(json.dumps(dataset, indent=4))
  f.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='build.py')

  parser.add_argument('-train_file', required=True,
                      help="Path to the training source data")
  parser.add_argument('-valid_file', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-test_file', required=True,
                      help="Path to the test source data")
  parser.add_argument('-train_num', type=int, default=100000,
                      help="No. of Training examples")
  parser.add_argument('-valid_num', type=int, default=2000,
                      help="No. of Validation examples")

  parser.add_argument('-output_folder', required=True,
                      help="Output folder for the prepared data")
  opt = parser.parse_args()
  print(opt)

  try:
    os.makedirs(opt.output_folder)
  except:
    pass

  processFiles(opt.train_file, opt.output_folder + '/train', "train", opt.train_num)
  processFiles(opt.valid_file, opt.output_folder + '/valid', "valid", opt.valid_num)
  processFiles(opt.test_file, opt.output_folder + '/test', "test", opt.valid_num)
