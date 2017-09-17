# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 11:57:34 2017

@author: Administrator
"""

import os
import pickle
from collections import Counter
import tensorflow as tf


def counts():
    cache = 'counter.pickle'
    if os.path.exists(cache):
      with open(cache, 'rb') as f:
        return pickle.load(f)
    directories = ['CBTest_data_P/training/', 'CBTest_data_P/validation/', 'CBTest_data_P/test/']
    files = [directory + file_name for directory in directories for file_name in os.listdir(directory)]
    counter = Counter()
    for file_name in files:
        with open(file_name, 'r') as f:
          lines = f.readlines()
          document = []
          query = []
          answer = []
          for line in lines:
              if line == '\n':
                  document = []
                  query = []
                  answer = []
              else:                  
                  words = line.split()    
                  if words[0] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']:                  
                      document.extend(words[1:])
                  elif words[0] == '21' :
                      query = line.split('\t')[0].split()[1:]                      
                      answer = line.split('\t')[1].split()
                      for token in document + query + answer:
                        counter[token] += 1
    with open(cache, 'wb') as f:
        pickle.dump(counter, f)
    
    return counter

def tokenize(index, word):
  #directories = ['cnn/questions/training/', 'cnn/questions/validation/', 'cnn/questions/test/']
  directories = ['CBTest_data_P/training/', 'CBTest_data_P/validation/', 'CBTest_data_P/test/']
  for directory in directories:
    out_name = directory.split('/')[-2] + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(out_name)
    files = map(lambda file_name: directory + file_name, os.listdir(directory))
    for file_name in files:       
      with open(file_name, 'r') as f:
          lines = f.readlines()
          document = []
          query = []
          answer = []
          for line in lines:
              if line == '\n':
                  document = []
                  query = []
                  answer = []
              else:                  
                  words = line.split()    
                  if words[0] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']:                  
                      document.extend(words[1:])
                  elif words[0] == '21' :
                      query = line.split('\t')[0].split()[1:]                      
                      answer = line.split('\t')[1].split()
                      
                      documents = [index[token] for token in document]
                      querys = [index[token] for token in query]
                      answers = [index[token] for token in answer]
                      example = tf.train.Example(
                         features = tf.train.Features(
                           feature = {
                             'document': tf.train.Feature(
                               int64_list=tf.train.Int64List(value=documents)),
                             'query': tf.train.Feature(
                               int64_list=tf.train.Int64List(value=querys)),
                             'answer': tf.train.Feature(
                               int64_list=tf.train.Int64List(value=answers))
                             }))
                      serialized = example.SerializeToString()
                      writer.write(serialized)
      
      
      



def main():
  counter = counts()
  print('num words',len(counter))  
  word, _ = zip(*counter.most_common())
  index = {token: i for i, token in enumerate(word)}
  tokenize(index, word)
  print('DONE')
  
  
if __name__ == "__main__":
  main()