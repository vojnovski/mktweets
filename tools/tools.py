#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 27 Apr 2013

@author: Viktor
'''

import pymongo
import twitter
import sys
from crawl import crawl
#import pylab as pl
#import numpy as np
#from bson.son import SON


def nr_of_new_friends(users, newuser, api):
    newfriends = set(crawl.getFriendsRateLimited(newuser, api))
    oldfriends = set([])
    for user in users:
        oldfriends = oldfriends | set(crawl.getFriendsRateLimited(user, api))
    print len(newfriends - oldfriends)


def prepare_api():
    try:
        fajl = open("twitterkey")
    except IOError:
        print >> sys.stderr, 'Error, Can not open file twitterkey containing the twitter api access keys'
        return
    keys = fajl.readline().split()
    return twitter.Api(consumer_key=keys[0], consumer_secret=keys[1], access_token_key=keys[2],
                       access_token_secret=keys[3])


def prepare_db():
    client = pymongo.MongoClient("localhost", 27017)
    return client.tweets


def max_method(api):
    for user in api.GetFriendIDs(screen_name='vojnovski'):
        api.GetUser(user_id=user)


def main():
    #db = prepare_db()
    api = prepare_api()
    #max_method(api)
    nr_of_new_friends(["Kontra991", "ljubopitna", "karamelicka", "bibliotekarot", "____Angie", "dragankucirov",
                       "_Stoimenov", "D_NazGUL", "vanilaafterdark", "m5zr", "kafedzika", "goranmitev", "bubu_mara",
                       "_Sagittarius8", "sazspasm", "letnikovski", "mandrak_zoro", "siontek", "strezovski",
                       "skopski_peder", "zkacarski"], "ramstoremall", api)
#    items = db.tweets.aggregate([
#        {"$group": {"_id": "$user.screen_name", "sum": {"$sum": 1}}}
#        ])
#
#    for item in items['result']:
#        print item

if __name__ == '__main__':
    main()
