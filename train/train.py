#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 27 Apr 2013

@author: Viktor
'''

import time
import pickle
import pymongo
import pylab as pl
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density


def load_tweets():
    """Load and return the tweets from the DB

    """
    print("Loading the data")
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    train_instances = 4
    n_samples = 7
    tweets_per_sample = 10000
    data = []

    #Manually tagged the first 120 users
    target = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
              1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    for i, user in enumerate(db.tweets.distinct('user.screen_name')[:n_samples]):
        first_tweet = db.tweets.find({'user.screen_name': user})[:1][0]
        sample = []
        sample.append(
            u"\n".join([tweet['text'] for tweet in db.tweets.find({'user.screen_name': user})[:tweets_per_sample]]))
        sample.append(first_tweet['user']['screen_name'])
        sample.append(first_tweet['user']['description'] if 'description' in first_tweet['user'] else "")
        sample.append(first_tweet['user']['name'])
        sample.append(target[i])
        data.append(sample)

    labels = target[:n_samples]

    data_train = data[:train_instances]
    data_test = data[train_instances:]
    y_train = labels[:train_instances]
    y_test = labels[train_instances:]

    print('Data loaded:')

    print("%d documents - (training set)" % (len(data_train)))
    print("%d documents - (test set)" % (len(data_test)))
    print("2 categories: [Male, Female]")
    print ""

    return data_train, data_test, y_train, y_test


def vectorize_tweets(data_train):
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()
    preprocess_tweets = lambda x: x[0]
    preprocess_screen_names = lambda x: x[1]
    preprocess_descriptions = lambda x: x[2]
    preprocess_names = lambda x: x[3]
    tweets_char_vect = TfidfVectorizer(preprocessor=preprocess_tweets, analyzer='char',
                                       ngram_range=(1, 5), strip_accents=None, charset_error='strict',
                                       sublinear_tf=True, max_df=0.5)
    tweets_word_vect = TfidfVectorizer(preprocessor=preprocess_tweets, analyzer='word',
                                       ngram_range=(1, 2), strip_accents=None, charset_error='strict',
                                       sublinear_tf=True, max_df=0.5)
    names_char_vect = TfidfVectorizer(preprocessor=preprocess_names, analyzer='char',
                                      ngram_range=(1, 5), strip_accents=None, charset_error='strict',
                                      sublinear_tf=True, max_df=0.5)
    names_word_vect = TfidfVectorizer(preprocessor=preprocess_names, analyzer='word', min_df=1,
                                      ngram_range=(1, 1), strip_accents=None, charset_error='strict',
                                      sublinear_tf=True, max_df=0.5)
    descriptions_char_vect = TfidfVectorizer(preprocessor=preprocess_descriptions, analyzer='char',
                                             ngram_range=(1, 5), strip_accents=None, charset_error='strict',
                                             sublinear_tf=True, max_df=0.5)
    descriptions_word_vect = TfidfVectorizer(preprocessor=preprocess_descriptions, analyzer='word',
                                             ngram_range=(1, 2), strip_accents=None, charset_error='strict',
                                             sublinear_tf=True, max_df=0.5)
    screen_names_char_vect = TfidfVectorizer(preprocessor=preprocess_screen_names, analyzer='char',
                                             ngram_range=(1, 5), strip_accents=None, charset_error='strict',
                                             sublinear_tf=True, max_df=0.5)
    #tweets_features = FeatureUnion([("tweets_char", tweets_char_vect), ("tweets_word", tweets_word_vect)])
    #names_features = FeatureUnion([("names_char", names_char_vect), ("names_word", names_word_vect)])
    #descriptions_features = FeatureUnion([("descriptions_char", descriptions_char_vect),
    #                                      ("descriptions_word", descriptions_word_vect)])
    tweets_all_features = FeatureUnion([("tweets_char", tweets_char_vect), ("tweets_word", tweets_word_vect),
                                        ("names_char", names_char_vect), ("names_word", names_word_vect),
                                        ("descriptions_char", descriptions_char_vect),
                                        ("descriptions_word", descriptions_word_vect),
                                        ("screen_names_char", screen_names_char_vect)])
    tweets_combined_features = tweets_all_features
    X_train = tweets_combined_features.fit_transform(data_train)
    duration = time.time() - t0
    print("done in %fs" % (duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print ""
    return tweets_combined_features, X_train


def extract_test_features(data_test, vectorizer):
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time.time()
    X_test = vectorizer.transform(data_test)
    duration = time.time() - t0
    print("done in %fs" % (duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print ""
    return X_test


def load_users(db, users):
    tweets = []
    for user in users:
        tweets.append(u"\n".join([tweet['text'] for tweet in db.tweets.find({'user.screen_name': user})]))
    return tweets


def classify_users(clf, vectorizer, users, tweets):
    X_new = vectorizer.transform(tweets)
    predicted = clf.predict(X_new)
    for doc, category in zip(users, predicted):
        print '%r => %s' % (doc, 'Female' if category == 0 else 'Male')


def naive_classify_unknown(X_train, y_train, vectorizer):
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    test_users = db.tweets.distinct('user.screen_name')
    classify_users(clf, vectorizer, test_users, load_users(db, test_users))


def benchmark(clf, X_train, X_test, y_train, y_test, feature_names):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("density: %f" % density(clf.coef_))
        if feature_names is not None:
            print("top 10 keywords per class:")
            if clf.coef_.shape[0] == 1:
                top10female = np.argsort(clf.coef_[0])[-10:]
                top10male = np.argsort(clf.coef_[0])[:10]
            else:
                top10female = np.argsort(clf.coef_)[-10:]
                top10male = np.argsort(clf.coef_)[:10]
            print("%s: %s" % ("Female", ", ".join(feature_names[top10female])))
            print("%s: %s" % ("Male", ", ".join(feature_names[top10male])))

        print ""

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                        target_names=['Female', 'Male']))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print ""
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


def test_classifiers(X_train, X_test, y_train, y_test, feature_names):
    results = []
    for clf, name in ((RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                      (Perceptron(n_iter=50), "Perceptron"),
                      (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                      (KNeighborsClassifier(n_neighbors=10), "kNN")):
        print '=' * 80
        print name
        results.append(benchmark(clf, X_train, X_test, y_train, y_test, feature_names))
    for penalty in ["l2", "l1"]:
        print '=' * 80
        print "%s penalty" % penalty.upper()
        # Train Liblinear model
        results.append(benchmark(LinearSVC(
            loss='l2', penalty=penalty, dual=False, tol=1e-3), X_train, X_test, y_train, y_test, feature_names))
    # Train SGD model
    results.append(benchmark(SGDClassifier(
        alpha=.0001, n_iter=50, penalty=penalty), X_train, X_test, y_train, y_test, feature_names))
    # Train SGD with Elastic Net penalty
    print '=' * 80
    print "Elastic-Net penalty"
    results.append(benchmark(SGDClassifier(
        alpha=.0001, n_iter=50, penalty="elasticnet"), X_train, X_test, y_train, y_test, feature_names))
    # Train NearestCentroid without threshold
    print '=' * 80
    print "NearestCentroid (aka Rocchio classifier)"
    results.append(benchmark(NearestCentroid(), X_train, X_test, y_train, y_test, feature_names))
    # Train sparse Naive Bayes classifiers
    print '=' * 80
    print "Naive Bayes"
    results.append(benchmark(MultinomialNB(alpha=.01), X_train, X_test, y_train, y_test, feature_names))
    results.append(benchmark(BernoulliNB(alpha=.01), X_train, X_test, y_train, y_test, feature_names))
    print '=' * 80
    print "LinearSVC with L1-based feature selection"
    results.append(benchmark(L1LinearSVC(), X_train, X_test, y_train, y_test, feature_names))
    return results


def show_plot(results):
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(4)]
    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)
    pl.figure(figsize=(12, 10))
    pl.title("Score")
    pl.barh(indices, score, .2, label="score", color='r')
    pl.barh(indices + .3, training_time, .2, label="training time", color='g')
    pl.barh(indices + .6, test_time, .2, label="test time", color='b')
    pl.yticks(())
    pl.legend(loc='best')
    pl.subplots_adjust(left=.25)
    pl.subplots_adjust(top=.95)
    pl.subplots_adjust(bottom=.05)
    for i, c in zip(indices, clf_names):
        pl.text(-.3, i, c)
    pl.show()


def train_test():
    #Load and save tweets to file
    data_train, data_test, y_train, y_test = load_tweets()
    pickle.dump((data_train, data_test, y_train, y_test), open("corpus", "wb"))
    # Vectorize training data
    vectorizer, X_train = vectorize_tweets(data_train)
    # Vectorize testing data
    X_test = extract_test_features(data_test, vectorizer)
    # Get extracted features
    feature_names = np.asarray(vectorizer.get_feature_names())
    # Do a fast Naive Bayes classification of all the users in the db. Can be used as a benchmark compariseon
    #naive_classify_unknown(X_train, y_train, vectorizer)
    #Do a set of tests on various classifiers
    results = test_classifiers(X_train, X_test, y_train, y_test, feature_names)
    # Show comparison plot
    show_plot(results)


def main():
    start_time = time.time()
    train_test()
    print "Time of Total execution: ", time.time() - start_time, "seconds"


if __name__ == '__main__':
    main()
