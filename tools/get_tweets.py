#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy
import os

# enter your Twitter API credentials
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = ""
access_token = ""
access_token_secret = ""

# username to look up
username = input("Enter username: ")


def get_all_tweets(username):
    # authorise twitter, initialise tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # initialise a list to hold all the tweets
    all_tweets = []

    # make initial request for most recent tweets (200 is the max allowed count)
    new_tweets = api.user_timeline(
        screen_name=username, count=200, tweet_mode="extended"
    )

    # save most recent tweets
    all_tweets.extend(new_tweets)

    # save the id of the last tweet id less one, to be used in max_id, which returns
    # only statuses with an ID less than (that is, older than) or equal to the specified ID
    last = all_tweets[-1].id - 1

    # keep fetching tweets till there's no more left
    while len(new_tweets) > 0:
        print(f"getting tweets before id %s" % last)

        new_tweets = api.user_timeline(
            screen_name=username, count=200, tweet_mode="extended", max_id=last
        )

        all_tweets.extend(new_tweets)

        last = all_tweets[-1].id - 1

    output = [tweet.full_text for tweet in all_tweets]

    # store output in text file
    with open(f"%s_tweets.txt" % username, "w") as f:
        for tweet in output:
            f.write("%s\n" % tweet)

    pass


if __name__ == "__main__":
    get_all_tweets(username)
