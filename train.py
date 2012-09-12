import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.base import BaseEstimator, clone
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features import TextFeatureTransformer, BadWordCounter, FeatureStacker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, chi2


from util import load_data, write_test, load_test

from IPython.core.debugger import Tracer


tracer = Tracer()


class BaggingClassifier(BaseEstimator):
    def __init__(self, estimator, n_estimators=10):
        self.estimator = estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators = []
        cv = ShuffleSplit(X.shape[0], n_iterations=self.n_estimators,
                test_size=0.3, indices=True)
        for train, test in cv:
            est = clone(self.estimator)
            est.fit(X[train], y[train])
            self.estimators.append(est)
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for est in self.estimators:
            probs += est.predict_proba(X)
        return probs / self.n_estimators


def simple_model():
    comments, dates, labels = load_data()
    select = SelectPercentile(score_func=chi2, percentile=16)

    clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
    countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            analyzer="char", binary=False)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            analyzer="word", binary=False)
    badwords = BadWordCounter()

    ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),
        ("words", countvect_word)])
    #ft = TextFeatureTransformer()
    pipeline = Pipeline([('vect', ft), ('select', select), ('logr', clf)])

    cv = ShuffleSplit(len(comments), n_iterations=5, test_size=0.2,
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

    tracer()


def grid_search():
    comments, dates, labels = load_data()
    param_grid = dict(logr__C=np.arange(1, 20),
            select__percentile=np.arange(2, 17, 1))
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=2)
    #countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            #analyzer="char", binary=False)
    #countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            #analyzer="word", binary=False)
    #badwords = BadWordCounter()

    #ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),
        #("words", countvect_word)])
    ft = TextFeatureTransformer()
    select = SelectPercentile(score_func=chi2)
    #select = LogisticRegression(tol=1e-8, penalty='l1')
    #select = RandomizedLogisticRegression(tol=1e-8, verbose=10,
            #n_resampling=20)
    #pipeline = Pipeline([('vect', ft), ('select', select), ('logr', clf)])
    pipeline = Pipeline([('select', select), ('logr', clf)])
    #pipeline = Pipeline([('vect', ft), ('logr', clf)])
    #pipeline = RFECV(estimator=clf, step=0.01, verbose=10, cv=5)
    cv = ShuffleSplit(len(comments), n_iterations=10, test_size=0.2)
    grid = GridSearchCV(pipeline, cv=cv, param_grid=param_grid, verbose=4,
            n_jobs=1, score_func=auc_score)
    X = ft.fit(comments).transform(comments)
    grid.fit(X, labels)
    print(grid.best_score_)
    print(grid.best_params_)

    tracer()
    cv_scores = grid.scores_
    for param in zip(cv_scores.params):
        means, errors = cv_scores.accumulate(param, 'max')
        plt.errorbar(cv_scores.values[param], means, yerr=errors)
        plt.set_xlabel(param)
        plt.set_ylim(0.6, 0.95)
        plt.ylim(0.85, 0.93)
        plt.savefig("grid_plot_%s.png" % param)
        plt.close()
    comments_test, dates_test = load_test()
    prob_pred = grid.best_estimator_.predict_proba(comments_test)
    write_test(prob_pred[:, 1])


def analyze_output():
    #from sklearn.feature_selection import SelectPercentile, chi2
    #from features import BadWordCounter
    comments, dates, labels = load_data()
    y_train, y_test, comments_train, comments_test = \
            train_test_split(labels, comments)
    #bad = BadWordCounter()
    #custom = bad.transform(comments_train)

    clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
    ft = TextFeatureTransformer().fit(comments_train, y_train)
    #countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            #analyzer="char", binary=False)
    #countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            #analyzer="word", binary=False)
    #badwords = BadWordCounter()

    #ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),
        #("words", countvect_word)])
    X_train = ft.transform(comments_train)
    #select = SelectPercentile(score_func=chi2, percentile=7)
    #X_train_s = select.fit_transform(X_train, y_train)
    clf.fit(X_train, y_train)
    X_test = ft.transform(comments_test)
    #X_test_s = select.transform(X_test)
    probs = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    print("auc: %f" % auc_score(y_test, probs[:, 1]))

    fp = np.where(pred > y_test)[0]
    fn = np.where(pred < y_test)[0]
    fn_comments = comments_test[fn]
    fp_comments = comments_test[fp]
    n_bad = X_test[:, -3].toarray().ravel()
    fn_comments = np.vstack([fn, n_bad[fn], probs[fn][:, 1], fn_comments]).T
    fp_comments = np.vstack([fp, n_bad[fp], probs[fp][:, 1], fp_comments]).T

    # visualize important features
    #important = np.abs(clf.coef_.ravel()) > 0.001
    #coef_ = select.inverse_transform(clf.coef_)
    coef_ = clf.coef_
    important = np.argsort(np.abs(coef_.ravel()))[-60:]
    feature_names = ft.get_feature_names()
    f_imp = feature_names[important]
    coef = coef_.ravel()[important]
    inds = np.argsort(coef)
    f_imp = f_imp[inds]
    coef = coef[inds]
    plt.plot(coef, label="l1")
    ax = plt.gca()
    ax.set_xticks(np.arange(len(coef)))
    labels = ax.set_xticklabels(f_imp)
    for label in labels:
        label.set_rotation(90)
    plt.savefig("ana.png")
    plt.show()

    tracer()


if __name__ == "__main__":
    #grid_search()
    #simple_model()
    analyze_output()
