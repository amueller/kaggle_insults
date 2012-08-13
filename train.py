import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features import TextFeatureTransformer, DensifyTransformer
from sklearn.metrics import auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#from sklearn.externals.joblib import Memory

#memory = Memory(cachdir="cache")

from time import strftime
from IPython.core.debugger import Tracer


tracer = Tracer()


def load_data():
    print("loading")
    comments = []
    dates = []
    labels = []

    #with codecs.open("train.csv", encoding="utf-8") as f:
    with open("train.csv") as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            labels.append(splitstring[0])
            dates.append(splitstring[1][:-1])
            comment = ",".join(splitstring[2:])
            comment = comment.strip().strip('"')
            comment.replace('_', ' ')
            comment = comment.replace("\\\\", "\\")
            comment = comment.decode('unicode-escape')
            comments.append(comment)
    labels = np.array(labels, dtype=np.int)
    dates = np.array(dates)
    return comments, dates, labels


def load_test():
    print("loading test set")
    comments = []
    dates = []

    with open("test.csv") as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            dates.append(splitstring[0][:-1])
            comment = ",".join(splitstring[1:])
            comment = comment.strip().strip('"')
            comment.replace('_', ' ')
            comment.replace('.', ' ')
            comment = comment.replace("\\\\", "\\")
            comment = comment.decode('unicode-escape')
            comments.append(comment)
    dates = np.array(dates)
    return comments, dates


def write_test(labels, fname=None):
    if fname is None:
        fname = "test_prediction_%s.csv" % strftime("%d_%H_%M")
    with open("test.csv") as f:
        with open(fname, 'w') as fw:
            fw.write(f.readline())
            for label, line in zip(labels, f):
                fw.write("%f," % label)
                fw.write(line)


def rfe_test():
    from sklearn.feature_selection import RFE
    comments, dates, labels = load_data()
    y_train, y_test, comments_train, comments_test = \
            train_test_split(labels, comments)
    clf = LogisticRegression(C=1, tol=1e-8)
    rfe = RFE(clf, step=1, n_features_to_select=12000)
    ft = TextFeatureTransformer(word_max_n=1).fit(comments_train)
    X_train = ft.transform(comments_train)
    rfe.fit(X_train, y_train)
    X_train_selected = rfe.transform(X_train)
    X_test = ft.transform(comments_test)
    X_test_selected = rfe.transform(X_test)
    clf.fit(X_train_selected, y_train)
    probs = clf.predict_proba(X_test_selected)
    print("auc: %f" % auc_score(y_test, probs[:, 1]))
    tracer()


def grid_search_feature_selection():
    comments, dates, labels = load_data()
    clf = LogisticRegression(tol=1e-8, C=0.5, penalty='l1')
    ft = TextFeatureTransformer(char=False, word_max_n=3)
    print("training and transforming for linear model")
    #X = ft.fit(comments).transform(comments)
    #X_ = clf.fit_transform(X, labels).toarray()
    print("training grid search")
    rf = RandomForestClassifier(n_estimators=200)
    dt = DensifyTransformer()
    pipeline = Pipeline([("features", ft), ("l1select", clf),
        ("make_dense", dt), ("rf", rf)])
    #param_grid = dict(features__word_max_n=np.arange(1, 3),
            #features__char=[True, False],
            #rf__max_depth=[20, 25, 30, 35],
            #rf__max_features=['log2', 'sqrt'],
            #rf__min_samples_leaf=[1],
            #l1select__C=[0.01, 0.1, 0.2, 0.5, 0.8, 1])
    param_grid = dict(features__word_max_n=[1],
            features__char=[False],
            rf__max_depth=[35, 40],
            rf__max_features=['log2', 10, 5, 8, 15],
            rf__min_samples_leaf=[1],
            l1select__C=[0.8])
    grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid, verbose=4,
            n_jobs=11, score_func=auc_score)
    grid.fit(comments, labels)
    tracer()
    comments_test, dates_test = load_test()
    prob_pred = grid.best_estimator_.predict_proba(comments_test)
    write_test(prob_pred[:, 1])


def grid_search():
    comments, dates, labels = load_data()
    #param_grid = dict(logr__C=2. ** np.arange(-6, 2), logr__penalty=['l1'],
            #vect__word_max_n=np.arange(1, 4), vect__char_max_n=[4],
            #vect__char_min_n=[3])
    param_grid = dict(logr__C=2. ** np.arange(0, 6))
    clf = LogisticRegression(tol=1e-8)
    ft = TextFeatureTransformer(char=False, word=False)
    pipeline = Pipeline([('vect', ft), ('logr', clf)])
    grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid, verbose=4,
            n_jobs=1, score_func=auc_score)

    grid.fit(comments, labels)
    print(grid.best_score_)
    print(grid.best_params_)
    comments_test, dates_test = load_test()
    prob_pred = grid.best_estimator_.predict_proba(comments_test)
    write_test(prob_pred[:, 1])
    tracer()


def analyze_output():
    comments, dates, labels = load_data()
    y_train, y_test, comments_train, comments_test = \
            train_test_split(labels, comments)
    #clf = LogisticRegression(tol=1e-8, penalty='l1', C=0.5)
    clf = LogisticRegression(tol=1e-8, penalty='l1', C=32)
    ft = TextFeatureTransformer(char=False, word=False, char_max_n=4,
            word_max_n=3, char_min_n=3).fit(comments_train)
    X_train = ft.transform(comments_train)
    clf.fit(X_train, y_train)
    X_test = ft.transform(comments_test)
    probs = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    print("auc: %f" % auc_score(y_test, probs[:, 1]))

    fp = np.where(pred > y_test)[0]
    fn = np.where(pred < y_test)[0]
    fn_comments = comments_test[fn]
    fp_comments = comments_test[fp]
    n_bad = X_test[:, -2].toarray().ravel()
    fn_comments = np.vstack([fn, n_bad[fn], probs[fn][:, 1], fn_comments]).T
    fp_comments = np.vstack([fp, n_bad[fp], probs[fp][:, 1], fp_comments]).T

    # visualize important features
    important = np.abs(clf.coef_.ravel()) > 0.001
    #important = np.abs(clf.coef_.ravel()) > 0.5
    feature_names = ft.get_feature_names()

    f_imp = feature_names[important]
    coef = clf.coef_.ravel()[important]
    inds = np.argsort(coef)
    f_imp = f_imp[inds]
    coef = coef[inds]
    plt.plot(coef)
    ax = plt.gca()
    ax.set_xticks(np.arange(len(coef)))
    labels = ax.set_xticklabels(f_imp)
    for label in labels:
        label.set_rotation(90)
    plt.show()

    tracer()


if __name__ == "__main__":
    #grid_search()
    analyze_output()
    #grid_search_feature_selection()
    #rfe_test()
