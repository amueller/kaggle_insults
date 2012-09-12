import numpy as np
from scipy import sparse
import re

import nltk
import nltk.collocations as col
import enchant
#from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.core.debugger import Tracer

tracer = Tracer()


def remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)


def build_tokenizer(func):
    regexp = re.compile(ur"\b\w\w+\b")
    tokenizer = lambda doc: [func(word) for word in
            regexp.findall(doc)]
    return tokenizer


def pos_tokenizer(doc):
    pass


class DensifyTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        return X


class BadWordCounter(BaseEstimator):
    def __init__(self):
        with open("my_badlist.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'allcaps', 'max_len',
            'mean_len', '@', '!', 'spaces', 'bad_ratio', 'n_bad',
            'capsratio'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        ## some handcrafted features!
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                                            for c in documents]
        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords_])
                                                for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, allcaps, max_word_len,
            mean_word_len, exclamation, addressing, spaces, bad_ratio, n_bad,
            allcaps_ratio]).T


class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, trans in self.transformer_list:
            features.append(trans.transform(X))
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out


def make_collocation_analyzer(collocations, length=2):
    def analyzer(document):
        cols = [bigram for bigram in nltk.ngrams(document, length)
                    if bigram in collocations]
        return cols

    return analyzer


class TextFeatureTransformer(BaseEstimator):
    def __init__(self):
        self.d = enchant.Dict("en_US")
        with open("my_badlist2.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

    def get_feature_names(self):
        feature_names = []
        feature_names.extend(self.unigram_vect.get_feature_names())
        feature_names.extend(self.bigram_vect_you.get_feature_names())
        feature_names.extend(self.trigram_vect_you.get_feature_names())
        feature_names.extend(self.pos_vect.get_feature_names())
        feature_names.extend(["n_nicks", "n_urls", "n_sentences",
            "n_non_words", "idiot", "moron"])
        feature_names.extend(['n_words', 'n_chars', 'toolong', 'allcaps',
            'max_len', 'mean_len', 'dots', '!', '?', 'spaces', 'bad_ratio',
            'n_bad', 'capsratio'])
        feature_names = [" ".join(w) if isinstance(w, tuple) else w
                            for w in feature_names]
        return np.array(feature_names)

    def fit(self, comments, y=None):
        designed_features, flat_words_lower, filtered_words, tags = \
                self._preprocess(comments)

        empty_analyzer = lambda x: x
        self.unigram_vect = TfidfVectorizer(analyzer=empty_analyzer, min_df=3)
        print("vecorizing")
        self.unigram_vect.fit(flat_words_lower)

        # pos tag vectorizer
        self.pos_vect = TfidfVectorizer(analyzer=empty_analyzer).fit(tags)

        # get the google bad word list
        #with open("google_badlist.txt") as f:
        self.bigram_measures = col.BigramAssocMeasures()
        self.trigram_measures = col.TrigramAssocMeasures()

        # extract bigram collocations including "you" (and your?)
        #col.BigramCollocationFinder.from_words([w for c in flat_words_lower
        #for w in c], window_size=4)

        col_you_bi = col.BigramCollocationFinder.from_documents(
                flat_words_lower)
        col_you_bi.apply_freq_filter(3)
        col_you_bi._apply_filter(lambda x, y: np.all([w != "you" for w in x]))
        # < 400 of these
        self.you_bigrams = col_you_bi.nbest(self.bigram_measures.chi_sq, 1000)
        self.col_you_bi = col_you_bi
        # make tfidfvectorizer that uses these bigrams
        self.bigram_vect_you = TfidfVectorizer(
                analyzer=make_collocation_analyzer(self.you_bigrams), min_df=3)
        self.bigram_vect_you.fit(flat_words_lower)

        # extract trigram collocations
        col_you_tri = col.TrigramCollocationFinder.from_documents(
                flat_words_lower)
        col_you_tri.apply_freq_filter(3)
        col_you_tri._apply_filter(lambda x, y: np.all([w != "you" for w in x]))
        # < 400 of these, too
        self.you_trigrams = col_you_tri.nbest(self.trigram_measures.chi_sq,
                                              1000)
        self.col_you_tri = col_you_tri
        self.trigram_vect_you = TfidfVectorizer(
            analyzer=make_collocation_analyzer(self.you_trigrams, 3), min_df=3)
        self.trigram_vect_you.fit(flat_words_lower)
        # make tfidfvectorizer that uses these bigrams

        return self

    def _preprocess(self, comments):
        # remove nicknames:
        nick = re.compile(ur"@\w\w+:?")
        url = re.compile(ur"http[^\s]*")

        n_nicks = [len(nick.findall(c)) for c in comments]
        comments_nonick = [nick.sub('', c) for c in comments]

        n_urls = [len(url.findall(c)) for c in comments_nonick]
        comments_nourl = [url.sub('', c) for c in comments_nonick]
        comments_ascii = [c.replace(u'\xa0', ' ') for c in comments_nourl]
        comments_ascii = [remove_non_ascii(c) for c in comments_ascii]
        ur = "you are "
        comments_ascii = [re.sub(ur"you'? ?a?re ", ur, c)
                for c in comments_ascii]
        idiot = [len(re.findall("you.? [\w ]* idi.t", c))
                for c in comments_ascii]
        moron = [len(re.findall("you.? [\w ]* m.r.n", c))
                for c in comments_ascii]
        # split into sentences
        sentences = [nltk.sent_tokenize(comment)
                     for comment in comments_ascii]
        # remove dots as they are annoying
        sentences = [[s.replace(".", " ") for s in sent] for sent in
                sentences]
        punctuation = \
            ['...', '.', '?', '!', ',', "''", '``', '#', '$', "'", "%", "&"]
        n_sentences = [len(sent) for sent in sentences]
        words = [[nltk.word_tokenize(s) for s in sent] for sent in sentences]
        tagged = [[nltk.pos_tag(s) for s in comment] for comment in words]
        tags = [[tag[1] for sent in comment for tag in sent]
                for comment in tagged]
        # remove ' <- this thing
        words = [[[w.strip("'") for w in s if w.strip("'")] for s in sent]
                for sent in words]
        filtered_words = [[w for sent in sents for w in sent
           if not w in punctuation]
           for sents in words]
        # detect weird stuff so we can spellcheck
        non_words = [[a for a in s if not self.d.check(a)]
                      for s in filtered_words]
        non_words = [[a for a in s if not nltk.corpus.wordnet.synsets(a)]
                    for s in non_words]
        non_words = [[a for a in s if not a.lower() in self.badwords_]
                    for s in non_words]
        n_non_words = [len(w) for w in non_words]
        flat_words_lower = [[w.lower() for w in comment]
                for comment in filtered_words]
        #flat = [a for s in non_words for a in s]
        #bla, blub = np.unique(flat, return_inverse=True)
        # not words, only there once. we could try and guess?
        #to_replace = bla[np.bincount(blub) == 1].tolist()
        features = [n_nicks, n_urls, n_sentences, n_non_words, idiot, moron]
        return [features, flat_words_lower,
                filtered_words, tags]

    def transform(self, comments):
        designed, flat_words_lower, filtered_words, tags = \
                self._preprocess(comments)

        # get started with real features:
        unigrams = self.unigram_vect.transform(flat_words_lower)
        you_bigrams = self.bigram_vect_you.transform(flat_words_lower)
        you_trigrams = self.trigram_vect_you.transform(flat_words_lower)
        pos_unigrams = self.pos_vect.transform(tags)

        ## some handcrafted features!
        n_words = [len(c) for c in filtered_words]
        n_chars = [len(c) for c in comments]
        too_long = np.array(n_chars) > 1000
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for sentence in comment
                           for w in sentence])
                   for comment in filtered_words]
        # longest word
        # after removeing all the stuff above, the comment migh be empty
        max_word_len = [np.max([len(w) for w in c])
                if len(c) else 0 for c in flat_words_lower]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c])
                if len(c) else 0 for c in flat_words_lower]

        # number of google badwords:
        # also take plurals
        n_bad = [np.sum([c.count(w) + c.count(w + "s")
                 for w in self.badwords_])
                 if len(c) else 0 for c in flat_words_lower]
        exclamation = [c.count("!") for c in comments]
        question = [c.count("?") for c in comments]
        spaces = [c.count(" ") for c in comments]
        dots = [c.count("...") for c in comments]

        # avoid nans
        allcaps_ratio = (np.array(allcaps)
                / (np.array(n_words, dtype=np.float) + 0.1))
        bad_ratio = np.array(n_bad) / (np.array(n_words, dtype=np.float) + 0.1)

        designed.extend([n_words, n_chars,
            allcaps, too_long, max_word_len, mean_word_len, dots, exclamation,
            question, spaces, bad_ratio, n_bad, allcaps_ratio])
        designed = np.array(designed).T

        features = []
        features.append(unigrams)
        features.append(you_bigrams)
        features.append(you_trigrams)
        features.append(pos_unigrams)
        features.append(sparse.csr_matrix(designed))
        features = sparse.hstack(features)

        return features.tocsr()
