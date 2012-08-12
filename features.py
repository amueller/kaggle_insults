import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


class TextFeatureTransformer(BaseEstimator):
    def __init__(self, word_max_n=2, char_min_n=1, char_max_n=6, char=False):
        self.word_max_n = word_max_n
        self.char_min_n = char_min_n
        self.char_max_n = char_max_n
        self.char = char

    def get_feature_names(self):
        if self.char:
            vcs = [self.countvect, self.countvect_char]
        else:
            vcs = [self.countvect]
        feature_names = [vc.get_feature_names() for vc in vcs]
        feature_names.append(['n_words', 'n_chars', 'allcaps', '@', '!',
            'max_len', 'mean_len', 'n_bad'])
        feature_names = np.hstack(feature_names)
        return feature_names

    def fit(self, comments, y=None):
        # get the google bad word list
        with open("google_badlist.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

        print("vecorizing")
        countvect = CountVectorizer(max_n=self.word_max_n, binary=True)
        countvect.fit(comments)
        self.countvect = countvect

        if self.char:
            countvect_char = CountVectorizer(max_n=self.char_max_n,
                    min_n=self.char_min_n, analyzer="char", binary=True)
            countvect_char.fit(comments)
            self.countvect_char = countvect_char
        return self

    def transform(self, comments):
        counts = self.countvect.transform(comments).tocsr()

        ## some handcrafted features!
        n_words = [len(c.split()) for c in comments]
        n_chars = [len(c) for c in comments]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in comments]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in comments]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                                            for c in comments]
        # number of google badwords:
        n_bad = [np.sum([w in c.lower() for w in self.badwords_])
                                                for c in comments]
        exclamation = [c.count("!") for c in comments]
        addressing = [c.count("@") for c in comments]

        features = np.array([n_words, n_chars, allcaps, max_word_len,
            mean_word_len, exclamation, addressing, n_bad]).T

        if self.char:
            counts_char = self.countvect_char.transform(comments).tocsr()
            features = sparse.hstack([counts, counts_char, features])
        else:
            features = sparse.hstack([counts, features])

        return features.tocsr()
