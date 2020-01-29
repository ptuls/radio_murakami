#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tweepy

from tweepy.error import TweepError

# username to look up
username = input("Enter username: ")

# max number of tweets allowed to fetch
MAX_TWEET_FETCH_COUNT = 200


def get_all_tweets(username):
    # enter your Twitter API credentials
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    # authorise twitter, initialise tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # initialise a list to hold all the tweets
    all_tweets = []
    new_tweets = []
    last_id = None
    iteration = 0

    while new_tweets or iteration == 0:
        print(f"getting tweets before id {last_id}")

        # make initial request for most recent tweets (200 is the max allowed count)
        try:
            new_tweets = api.user_timeline(
                screen_name=username,
                count=MAX_TWEET_FETCH_COUNT,
                tweet_mode="extended",
                max_id=last_id,
            )
        except TweepError:
            print("Username {username} doesn't exist")
            break

        if not new_tweets:
            print(f"Username {username} has not published any tweets")
            break

        # save most recent tweets
        all_tweets.extend([tweet.full_text for tweet in new_tweets])

        # save the id of the last tweet id less one, to be used in max_id, which returns
        # only statuses with an ID less than (that is, older than) or equal to the specified ID
        last_id = new_tweets.max_id - 1
        iteration += 1

    return all_tweets


def write_output(all_tweets):
    # store output in text file
    with open(f"{username}_tweets.txt", "w") as f:
        for tweet in all_tweets:
            f.write(tweet + "\n")


def main(username):
    all_tweets = get_all_tweets(username)
    if all_tweets:
        write_output(all_tweets)


if __name__ == "__main__":
    main(username)
