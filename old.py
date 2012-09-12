import numpy as np

from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features import BadWordCounter, FeatureStacker
#from features import BadWordCounter, FeatureStacker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc_score
from sklearn.feature_selection import SelectPercentile, chi2

from util import load_data

from IPython.core.debugger import Tracer


tracer = Tracer()


def jellyfish():
    #import jellyfish

    comments, dates, labels = load_data()
    y_train, y_test, comments_train, comments_test = \
            train_test_split(labels, comments)
    #ft = TextFeatureTransformer(word_range=(1, 1),
            #tokenizer_func=jellyfish.porter_stem).fit(comments_train)
    #ft = TextFeatureTransformer(word_range=(1, 1),
            #tokenizer_func=None).fit(comments_train)
    tracer()
    countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            analyzer="char", binary=False)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            analyzer="word", binary=False)
    badwords = BadWordCounter()
    ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),
        ("words", countvect_word)])
    clf = LogisticRegression(C=1, tol=1e-8)
    X_train = ft.transform(comments_train)
    clf.fit(X_train, y_train)
    X_test = ft.transform(comments_test)
    probs = clf.predict_proba(X_test)
    print("auc: %f" % auc_score(y_test, probs[:, 1]))


def test_stacker():
    comments, dates, labels = load_data()
    clf = LogisticRegression(tol=1e-8, C=0.01, penalty='l2')
    countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            analyzer="char", binary=False)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            analyzer="word", binary=False)
    badwords = BadWordCounter()
    select = SelectPercentile(score_func=chi2)
    char_select = Pipeline([('char_count', countvect_char),
                            ('select', select)])
    words_select = Pipeline([('word_count', countvect_word),
                             ('select', select)])
    badwords_select = Pipeline([('badwords', badwords), ('select', select)])

    stack = FeatureStacker([("badwords", badwords_select),
                            ("chars", char_select),
                            ("words", words_select)])
    #stack.fit(comments)
    #features = stack.transform(comments)

    #print("training and transforming for linear model")
    print("training grid search")
    pipeline = Pipeline([("features", stack), ("clf", clf)])
    param_grid = dict(clf__C=[0.31, 0.42, 0.54],
                      features__words__select__percentile=[5, 7])
    grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid, verbose=4,
           n_jobs=1, score_func=auc_score)
    grid.fit(comments, labels)
    tracer()
    #comments_test, dates_test = load_test()
    #prob_pred = grid.best_estimator_.predict_proba(comments_test)
    #write_test(prob_pred[:, 1])


def bagging():
    from sklearn.feature_selection import SelectPercentile, chi2

    comments, dates, labels = load_data()
    select = SelectPercentile(score_func=chi2, percentile=4)

    clf = LogisticRegression(tol=1e-8, penalty='l2', C=7)
    #clf = BaggingClassifier(logr, n_estimators=50)
    countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            analyzer="char", binary=False)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            analyzer="word", binary=False)
    badwords = BadWordCounter()

    ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),
        ("words", countvect_word)])
    #ft = TextFeatureTransformer()
    pipeline = Pipeline([('vect', ft), ('select', select), ('logr', clf)])

    cv = ShuffleSplit(len(comments), n_iterations=20, test_size=0.2,
            indices=True)
    scores = []
    for train, test in cv:
        X_train, y_train = comments[train], labels[train]
        X_test, y_test = comments[test], labels[test]
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)
        scores.append(auc_score(y_test, probs[:, 1]))
        print("score: %f" % scores[-1])
    print(np.mean(scores), np.std(scores))
