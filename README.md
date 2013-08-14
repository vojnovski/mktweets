
Summary:
--------

mktweets is an attempt to construct a gender classifier for Macedonian twitter users.


Installing:
-----------
First, the following dependencies need to be satisfied:
  * nltk
  * python-twitter
  * mongodb + pymongo
  * guess_language from https://bitbucket.org/spirit/guess_language
  * numpy
  * scipy
  * matplotlib
  * pylab
  * scikit_learn

Usage:
------
In order to crawl tweets and make a dataset to learn from, put the twitter api key in the
module folder named *twitterkey*, change the mongodb ip:port as well as the starting username
in crawl/crawl.py, run the mongodb server and then start crawl/crawl.py

In order to train the module, change the number of records used for training/test in
train/train.py as well as hand-tag them in the same file, and then run it. Output is in console
as a comparison chart between different methods.

