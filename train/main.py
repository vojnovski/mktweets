#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 27 Apr 2013

@author: Viktor
'''

import time
import nltk
import twitter
import pickle
from guess_language import guess_language
import sys, codecs, locale
import pymongo
import json

def preprocess(db):
    for tweet in db.tweets:
        pass
        # if tweet in Macedonian:        
    
def main():
    start_time = time.time()
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    db.tweets.ensure_index('id', unique = True, dropDups = True)
    merged_users = preprocess(db)
    
    print "Time of execution: ", time.time() - start_time, "seconds"
    
if __name__ == '__main__':
    main()
