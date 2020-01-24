# Radio Murakami

**Welcome to the Radio Murakami repository!**

Here, we develop a bot that spouts out quotes as good as the writings of [Haruki Murakami](https://en.wikipedia.org/wiki/Haruki_Murakami) himself. This is based off his writing style as well as quotes and interviews he has done in the past.

## Data

### Interviews & Short Essays
- [Jazz Messenger](https://www.nytimes.com/2007/07/08/books/review/Murakami-t.html?action=click&module=RelatedCoverage&pgtype=Article&region=Footer)
- [Haruki Murakami Says He Doesn’t Dream. He Writes.](https://www.nytimes.com/2018/10/10/books/murakami-killing-commendatore.html)
- [What’s Needed is Magic: Writing Advice from Haruki Murakami](https://lithub.com/whats-needed-is-magic-writing-advice-from-haruki-murakami/)
- [Haruki Murakami, The Art of Fiction No. 182](https://www.theparisreview.org/interviews/2/haruki-murakami-the-art-of-fiction-no-182-haruki-murakami)
- [An Unrealistic Dreamer](https://beb.mobi/2012/02/07/speaking-on-fukushima-an-unrealistic-dreamer-haruki-murakami/)
- [Haruki Murakami: The Moment I Became a Novelist](https://lithub.com/haruki-murakami-the-moment-i-became-a-novelist/#)
- [Always on the Side of the Egg](https://www.haaretz.com/israel-news/culture/1.5076881)
- [The Best of Haruki Murakami’s Advice Column](https://www.vulture.com/2015/02/best-of-haruki-murakami-advice-column.html)
- [A Conversation with Murakami about Sputnik Sweetheart](http://www.harukimurakami.com/resource_category/q_and_a/a-conversation-with-haruki-murakami-about-sputnik-sweetheart)
- [Questions for Murakami about Kafka on the Shore](http://www.harukimurakami.com/resource_category/q_and_a/questions-for-haruki-murakami-about-kafka-on-the-shore)
- [The novelist in wartime](https://www.salon.com/control/2009/02/20/haruki_murakami/)
- [Surreal often more real for author Haruki Murakami](https://www.reuters.com/article/us-books-author-murakami-idUSTRE5AO11720091125)
- [The Salon: Haruki Murakami](https://www.salon.com/control/1997/12/16/int_2/)
- [An Interview with Haruki Murakami](https://www.bookbrowse.com/author_interviews/full/index.cfm?author_number=1103)
- [When I Run I Am in a Peaceful Place](https://www.spiegel.de/international/world/spiegel-interview-with-haruki-murakami-when-i-run-i-am-in-a-peaceful-place-a-536608.html)
- [Haruki Murakami on Parallel Realities](https://www.newyorker.com/books/this-week-in-fiction/haruki-murakami-2018-09-03)

To be updated
- [Free Haruki Murakami Short Stories, Essays, Interviews, Speeches](https://bookoblivion.com/2016/12/05/free-haruki-muakami-short-stories-essays/)
- [All works by Haruki Murakami on The New Yorker](https://www.newyorker.com/contributors/haruki-murakami)

## Model

We fine tuned a GPT-2 and BERT model using the datasets above.

## Contributors

In this repository, we use [`pre-commit`](https://pre-commit.com/) to ensure consistency of formatting. To install for Mac, run
```
brew install pre-commit
```

Once installed, in the command line of the repository, run
```
pre-commit install
```
This will install `pre-commit` to the Git hook, so that `pre-commit` will run and fix files covered in its config before committing. 

## References

[1] Radford A., Wu J., Child R., Luan D., Amodei D., and Sutskever I., "Language Models are Unsupervised Multitask Learners", 2019. ([link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))

[2] Devlin J., Chang M-W., Lee K., and Toutanova K., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume I (Long and Short Papers), pp. 4171-4186, June 2019. ([link](https://www.aclweb.org/anthology/N19-1423/))

[3] Vaswani A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I, "Attention is all you need", Advances in Neural Information Processing Systems, pp. 5998–6008, 2017. ([link](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
