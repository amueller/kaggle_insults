import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
#from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
import scipy
#from sklearn.ensemble import RandomForestClassifier
from IPython.core.debugger import Tracer

tracer = Tracer()


def load_data():
    print("loading")
    comments = []
    dates = []
    labels = []

    with open("train.csv") as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            labels.append(splitstring[0])
            dates.append(splitstring[1][:-1])
            comment = ",".join(splitstring[2:])
            comment = comment.strip().strip('"')
            comment.replace('_', ' ')
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
            comments.append(comment)
    dates = np.array(dates)
    return comments, dates


def write_test(labels, fname="test_prediction.csv"):
    with open("test.csv") as f:
        with open(fname, 'w') as fw:
            fw.write(f.readline())
            for label, line in zip(labels, f):
                fw.write("%f," % label)
                fw.write(line)


def grid_search():
    comments, dates, labels = load_data()

    print("vecorizing")
    countvect = CountVectorizer(max_n=2)
    countvect_char = CountVectorizer(max_n=6, analyzer="char")
    #countvect = TfidfVectorizer()

    counts = countvect.fit_transform(comments)
    counts_char = countvect_char.fit_transform(comments)
    features = scipy.sparse.hstack([counts, counts_char])
    tracer()
    #counts_array = counts.toarray()
    #indicators = sparse.csr_matrix((counts_array > 0).astype(np.int))

    #X_train, X_test, y_train, y_test = train_test_split(counts, labels,
                                                        #test_size=0.5)
    #inds = np.random.permutation(len(labels))
    #n_samples = len(labels)
    #print("training")
    #param_grid = dict(logr__C=2. ** np.arange(-6, 4),
            #logr__penalty=['l1', 'l2'],
            #vect__max_n=np.arange(1, 4), vect__lowercase=[True, False])
    param_grid = dict(C=2. ** np.arange(-3, 4),
            penalty=['l1', 'l2'])
    #clf = LinearSVC(tol=1e-8, penalty='l1', dual=False, C=0.5)
    clf = LogisticRegression(tol=1e-8, penalty='l1', C=2)
    #pipeline = Pipeline([('vect', countvect), ('logr', clf)])
    #feature_selector.fit(comments, labels)
    #features = feature_selector.transform(comments).toarray()

    #clf = NearestCentroid()

    #param_grid = dict(max_depth=np.arange(1, 20), max_features=['sqrt', 'log2', None])
    #clf = RandomForestClassifier(n_estimators=10)

    grid = GridSearchCV(clf, cv=5, param_grid=param_grid, verbose=4,
            n_jobs=11)
    #print(cross_val_score(clf, counts, labels, cv=3))

    grid.fit(features, labels)
    print(grid.best_score_)
    print(grid.best_params_)
    #clf.fit(X_train, y_train)
    #print(clf.score(X_test, y_test))
    tracer()
    comments_test, dates_test = load_test()
    prob_pred = grid.best_estimator_.predict_proba(comments_test)
    write_test(prob_pred[:, 1])

if __name__ == "__main__":
    grid_search()
