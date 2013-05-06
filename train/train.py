#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 27 Apr 2013

@author: Viktor
'''

import time
import pickle
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB

def load_tweets(db, urls=True, mentions=True):
    """Load and return the tweets from the DB

    """
    n_samples = 24
    tweets_per_sample = 2000

    tweets = []
    screen_names= []
    descriptions = []
    names = []

    #Manually tagged the first 24 users
    target = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ,0 ,0 ,0 ,1 ,0 ,0]
    
    for user in db.tweets.distinct('user.screen_name')[:n_samples]:
        first_tweet = db.tweets.find({'user.screen_name': user})[:1][0]
        tweets.append(u"\n".join([tweet['text'] for tweet in db.tweets.find({'user.screen_name': user})[:tweets_per_sample]]))
        screen_names.append(first_tweet['user']['screen_name'])
        #descriptions.append(first_tweet['user']['description'])  # Some users seem to have no description
        names.append(first_tweet['user']['name'])
        #populate target[i]: where from?        
    
    return Bunch(tweets=tweets, screen_names=screen_names, descriptions=descriptions, names=names, target=target, 
                 target_names=['Female', 'Male'],
                 DESCR="Database tweets dump, agreggated over users",
                 )


def vectorize_tweets(corpus):
    tweets_char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,5), strip_accents=None, charset_error='strict')
    tweets_word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3), strip_accents=None, charset_error='strict')
    tweets_combined_features = FeatureUnion([("tweets_char", tweets_char_vectorizer), ("tweets_word", tweets_word_vectorizer)])  
    X = tweets_combined_features.fit_transform(corpus.tweets) 
    return tweets_combined_features, X


def train_classifier(X, corpus):
    clf = MultinomialNB().fit(X, corpus.target)
    return clf


def load_users(db, users):
    tweets = []
    for user in users:
        tweets.append(u"\n".join([tweet['text'] for tweet in db.tweets.find({'user.screen_name': user})]))
    return tweets


def test_classifier(clf, vectorizer, users, tweets):
    X_new = vectorizer.transform(tweets)
    predicted = clf.predict(X_new)
    for doc, category in zip(users, predicted):
        print '%r => %s' % (doc, 'Female' if category == 0 else 'Male')   


def main():
    start_time = time.time()
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    corpus = load_tweets(db, 'all')
    vectorizer, X = vectorize_tweets(corpus)
    clf = train_classifier(X,corpus)
    pickle.dump( (vectorizer, clf), open( "classifier", "wb" ) )
    #vectorizer, clf = pickle.load(open ("classifier", "rb"))
    test_users = ['vojnovski']
    test_classifier(clf, vectorizer, test_users, load_users(db, test_users))
    print "Time of execution: ", time.time() - start_time, "seconds"
    
    
if __name__ == '__main__':
    main()


