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

broj_na_iteracii = 2
projdeni = set([])

def getStatusiRateLimited(korisnik, api, maxid=-1):
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

def langPercentage(statusi, language='mk'):
    n = 0
    for status in statusi:
        if guess_language(unicode(status.text)) == language:            
            n=n+1
    if len(statusi) == 0:
        return 0
    return n/float(len(statusi))

def getSiteStatusi(korisnik, api, site=True):
    statusi = getStatusiRateLimited(korisnik, api)
    if langPercentage(statusi) < 0.1:
        return [], False
    print "Vkupno: " + str(len(statusi)) + " statusi od @" + korisnik + ": " + statusi[-1].text.encode('ascii', 'ignore')
    if not site:
        return statusi, True
    while True:
        newstatusi = getStatusiRateLimited(korisnik, api, getMinid(statusi)-1)
        if newstatusi != []:
            statusi += newstatusi
            print "Vkupno: " + str(len(statusi)) + "; Uste statusi od @" + korisnik + ": " + statusi[-1].text.encode('ascii', 'ignore')
        else:
            return statusi, True

def getFriendsRateLimited(korisnik, api):
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

def svrti(korisnik, iteracija, db, api, tree):
    if korisnik not in projdeni:
        statuses, mk = getSiteStatusi(korisnik, api,  True)
        snimi(statuses, db)
        projdeni.add(korisnik)
        if mk and not iteracija + 1 == broj_na_iteracii:
            for friend in getFriendsRateLimited(korisnik, api):
                    print "Nivo " + str(iteracija) + ": " + tree + "->" + friend.screen_name
                    svrti(friend.screen_name, iteracija+1, db, api, tree + "->" + friend.screen_name )
    else:
        print "Vekje projden: " + korisnik

def test():
    api = prepareApi()
    
def prepareApi():
    try:
        fajl = open("twitterkey")
    except IOError:
        print >> sys.stderr, 'Error, Can not open file twitterkey containing the twitter api access keys'
        return
    keys = fajl.readline().split()
    return twitter.Api(consumer_key=keys[0], consumer_secret=keys[1], access_token_key=keys[2], access_token_secret=keys[3])

def main():
    start_time = time.time()
    api = prepareApi()
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    db.tweets.ensure_index('id', unique = True, dropDups = True)
    tvitovi = svrti("vojnovski", 0, db, api, "vojnovski")
    print "Time of execution: ", time.time() - start_time, "seconds"
    
if __name__ == '__main__':
    main()
