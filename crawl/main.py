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
import guess_language
import sys, codecs, locale
import pymongo
import json


def getStatusiRateLimited(korisnik, maxid=-1): 
    time_between_retries = 5
    while True:
        try:
            if maxid == -1:
                statuses = api.GetUserTimeline(screen_name=korisnik, count=200, include_entities=True)
            else:
                statuses = api.GetUserTimeline(screen_name=korisnik, count=200, max_id = maxid, include_entities=True)
            return statuses
        except twitter.TwitterError, e:
            print e.message
            if not "Rate limit exceeded" in e.message:
                return []          
            time_between_retries *= 2
            time.sleep(time_between_retries)
   
def getMinid(statusi):
    return min([p.id for p in statusi])
    
def getSiteStatusi(korisnik, site=True):
    statusi = getStatusiRateLimited(korisnik)
    print "Vkupno: " + str(len(statusi)) + " statusi od @" + korisnik + ": " + statusi[-1].text.encode('ascii', 'ignore')
    if not site:
        return statusi
    while True:
        newstatusi = getStatusiRateLimited(korisnik, getMinid(statusi)-1)
        if newstatusi != []:
            statusi += newstatusi
            print "Vkupno: " + str(len(statusi)) + "; Uste statusi od @" + korisnik + ": " + statusi[-1].text.encode('ascii', 'ignore')
        else:
            return statusi
        
def getFriendsRateLimited(korisnik):
    time_between_retries = 5
    while True:
        try:
            friends = api.GetFriends(user=korisnik)
            return friends
        except twitter.TwitterError, e:
            print e.message
            if not "Rate limit exceeded" in e.message:
                return []          
            time_between_retries *= 2
            time.sleep(time_between_retries)
        
def snimi(statuses, db):
    for status in statuses:
        try:
            db.tweets.save(json.loads(status.AsJsonString()), check_keys=False)
        except:
            pass
            
def svrti(korisnik, iteracija, db):
    if korisnik not in projdeni:
        if not iteracija + 1 == broj_na_iteracii:   
            for friend in getFriendsRateLimited(korisnik):
                    print "Nivo " + str(iteracija) + ": " + korisnik + ": " + friend.screen_name
                    svrti(friend.screen_name, iteracija+1, db) 
        statuses = getSiteStatusi(korisnik, True)   
        snimi(statuses, db)
        projdeni.add(korisnik)  
    else:
        print "Vekje projden: " + korisnik

start_time = time.time()
print sys.stdout.encoding
file = open("twitterkey")
keys = file.readline().split()
api = twitter.Api(consumer_key=keys[0], consumer_secret=keys[1], access_token_key=keys[2], access_token_secret=keys[3])
broj_na_iteracii = 2
projdeni = set([])
client = pymongo.MongoClient("localhost", 27017)
db = client.tweets
db.tweets.ensure_index('id', unique = True, dropDups = True)
tvitovi = svrti("vojnovski", 0, db)
#print len(tvitovi)
#print [s.text for s in tvitovi]
#pass

#print guess_language(unicode("@jovanatozija Незнам зашо неќе да работи срањево"))
#print guess_language(unicode("hairy fucker"))
#print [guess_language(unicode(s.text)) for s in statuses]
#sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)
#tagged = nltk.pos_tag(tokens)
#print tagged[0:6]
