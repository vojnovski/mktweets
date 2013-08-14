#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 27 Apr 2013

@author: Viktor
'''

import time
import twitter
#import pickle
from guess_language import guess_language
import sys
import pymongo
import json
import datetime
import calendar

BROJ_NA_ITERACII = 4
projdeni = set([])
user_ids = dict()


def api_sleep(api, method):
    """ Spie odredeno vreme posle blokiracki povik.
    """
    try:
        limit_status = api.GetRateLimitStatus()
    except twitter.TwitterError:
        return
    dtcode = datetime.datetime.utcnow()
    unixtime = calendar.timegm(dtcode.utctimetuple())
    if method == 'Timeline':
        sleeptime = limit_status[u'resources'][u'statuses'][u'/statuses/user_timeline'][u'reset']
    elif method == 'User':
        sleeptime = limit_status[u'resources'][u'users'][u'/users/show/:id'][u'reset']
    else:
        sleeptime = limit_status[u'resources'][u'friends'][u'/friends/list'][u'reset']
    sleeptime = sleeptime - unixtime + 10
    print 'Spijam ', sleeptime, 'sekundi'
    time.sleep(sleeptime)


def get_statusi_ratelimited(korisnik, api, maxid=-1):
    """ Zema 200 statusi na korisnikot pocnuvajkji od id-to koe greska specificirano.
    """
    while True:
        try:
            if maxid == -1:
                statuses = api.GetUserTimeline(
                    screen_name=korisnik, count=200)
            else:
                statuses = api.GetUserTimeline(
                    screen_name=korisnik, count=200, max_id=maxid)
            return statuses
        except twitter.TwitterError, greska:
            if "Not authorized" in greska or not "Rate limit exceeded" in greska.message[0]['message']:
                print greska
                return []
            api_sleep(api, 'Timeline')


def lang_percentage(statusi, language='mk'):
    """ Vrakja procent na tvitovi koi se na odredeniot jazik.
    """
    n = 0
    for status in statusi:
        if guess_language(unicode(status.text)) == language:
            n = n + 1
    if len(statusi) == 0:
        return 0
    return n / float(len(statusi))


def in_db(statusi, db):
    if statusi:
        if statusi[0]:
            if db.tweets.find({"id": statusi[0].id}).count() == 1:
                return True
    return False


def get_site_statusi(korisnik, api, db, site=True):
    print "    Pocnuvam @" + korisnik
    statusi = get_statusi_ratelimited(korisnik, api)
    if lang_percentage(statusi) < 0.1:
        return [], False
    if in_db(statusi, db):
        return [], True
    print "     Vkupno " + str(len(statusi)) + " statusi od @" + korisnik + ": " + \
          statusi[-1].text.encode('ascii', 'ignore')
    if not site:
        return statusi, True
    while True:
        newstatusi = get_statusi_ratelimited(korisnik, api, min([p.id for p in statusi]) - 1)
        if in_db(newstatusi, db):
            return statusi, True
        if newstatusi != []:
            statusi += newstatusi
            print "    Vkupno: " + str(len(statusi)) + "; Uste statusi od @" + korisnik + ": " + \
                  statusi[-1].text.encode('ascii', 'ignore')
        else:
            return statusi, True


def get_user_ratelimited(user_id, api):
    if user_id in user_ids:
        return user_ids[user_id]
    while True:
        try:
            user = api.GetUser(user_id=user_id)
            user_ids[user_id] = user.screen_name
            return user.screen_name
        except twitter.TwitterError, greska:
            if "Not authorized" in greska or not "Rate limit exceeded" in greska.message[0]['message']:
                print greska
                return None
            api_sleep(api, "User")


def get_friends_ratelimited(korisnik, api):
    print "Gi zemam prijatelite na " + korisnik
    while True:
        try:
            friends = api.GetFriendIDs(screen_name=korisnik)
            return friends
        except twitter.TwitterError, greska:
            if "Not authorized" in greska or not "Rate limit exceeded" in greska.message[0]['message']:
                print greska
                return []
            api_sleep(api, "Friends")


def snimi(statuses, db):
    for status in statuses:
        try:
            db.tweets.save(json.loads(status.AsJsonString()), check_keys=False)
        except:
            #bad hack, should do a typed catch, but it works for now
            return


def svrti(korisnik, iteracija, db, api, tree):
    if korisnik not in projdeni:
        statuses, lang_is_mk = get_site_statusi(korisnik, api, db, True)
        snimi(statuses, db)
        projdeni.add(korisnik)
        if lang_is_mk and not iteracija + 1 == BROJ_NA_ITERACII:
            for friend_id in get_friends_ratelimited(korisnik, api):
                friend_name = get_user_ratelimited(friend_id, api)
                if friend_name is not None:
                    print "Nivo " + str(iteracija) + ": " + tree + "->" + friend_name
                    svrti(friend_name, iteracija + 1, db, api, tree + "->" + friend_name)
    else:
        print "    Vekje projden: @" + korisnik


def prepare_api():
    try:
        fajl = open("twitterkey")
    except IOError:
        print >> sys.stderr, 'Error, Can not open file twitterkey containing the twitter api access keys'
        return
    keys = fajl.readline().split()
    return twitter.Api(consumer_key=keys[0], consumer_secret=keys[1], access_token_key=keys[2],
                       access_token_secret=keys[3])


def main():
    start_time = time.time()
    api = prepare_api()
    client = pymongo.MongoClient("localhost", 27017)
    db = client.tweets
    db.tweets.ensure_index('id', unique=True, dropDups=True)
    svrti("vojnovski", 0, db, api, "vojnovski")
    print "Vreme na izvrsuvanje: ", time.time() - start_time, "sekundi"


if __name__ == '__main__':
    main()
