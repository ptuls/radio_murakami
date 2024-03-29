#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import tweepy

from collections import namedtuple
import csv
import sys
from typing import List
from tweepy.error import TweepError

# max number of tweets allowed to fetch
MAX_TWEET_FETCH_COUNT = 200


AccessConfig = namedtuple(
    "AccessConfig",
    "consumer_key consumer_secret access_token access_token_secret",
)


def get_access_config() -> AccessConfig:
    return AccessConfig(
        consumer_key=os.getenv("CONSUMER_KEY"),
        consumer_secret=os.getenv("CONSUMER_SECRET"),
        access_token=os.getenv("ACCESS_TOKEN"),
        access_token_secret=os.getenv("ACCESS_TOKEN_SECRET"),
    )


def get_all_tweets(
    username: str, config: AccessConfig, with_retweets: bool
) -> List[str]:
    # authorise twitter, initialise tweepy
    auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_token_secret)
    api = tweepy.API(auth)

    # initialise a list to hold all the tweets
    all_tweets = []
    new_tweets = []
    last_id = None
    first_iteration = True

    while new_tweets or first_iteration:
        print(f"getting tweets before id {last_id}")

        # make initial request for most recent tweets (200 is the max allowed count)
        try:
            new_tweets = api.user_timeline(
                screen_name=username,
                count=MAX_TWEET_FETCH_COUNT,
                tweet_mode="extended",
                max_id=last_id,
                include_rts=with_retweets,
            )
        except TweepError:
            print(f"Username {username} doesn't exist")
            break

        if not new_tweets:
            if first_iteration:
                print(f"Username {username} has not published any tweets")
            break

        # save most recent tweets
        all_tweets.extend([tweet.full_text for tweet in new_tweets])

        # save the ID of the last tweet ID less one, to be used in max_id, which returns
        # only statuses with an ID less than (that is, older than) or equal to the specified ID
        last_id = new_tweets.max_id - 1
        first_iteration = False

    print(f"tweets retrieved: {len(all_tweets)}")
    return all_tweets


def write_output(all_tweets: List[str], is_csv: bool) -> None:
    if is_csv:
        # create list of strings to populate the csv
        out_tweets = [[all_tweets[i]] for i in range(len(all_tweets))]

        # store output in csv file
        with open(f"{username}_tweets.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(out_tweets)
    else:
        # store output in txt file
        with open(f"{username}_tweets.txt", "w") as f:
            for tweet in all_tweets:
                f.write(tweet + "\n---\n")


def main(
    username: str, config: AccessConfig, is_csv: bool, with_retweets: bool
) -> None:
    all_tweets = get_all_tweets(username, config, with_retweets)

    if all_tweets:
        write_output(all_tweets, is_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get tweets from a user and write to desired file format."
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="outputs tweet formats to single column CSV",
    )
    parser.add_argument(
        "-r",
        "--retweets",
        action="store_true",
        help="output including retweets",
    )
    args = parser.parse_args()
    # username to look up
    username = input("Enter username: ")
    config = get_access_config()
    main(username, config, args.csv, args.retweets)
