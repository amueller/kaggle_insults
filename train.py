import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#from scipy import sparse
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
    countvect = CountVectorizer(max_n=1)
    #countvect = TfidfVectorizer()

    counts = countvect.fit_transform(comments)
    #counts_array = counts.toarray()
    #indicators = sparse.csr_matrix((counts_array > 0).astype(np.int))

    #X_train, X_test, y_train, y_test = train_test_split(counts, labels,
                                                        #test_size=0.5)
    #inds = np.random.permutation(len(labels))
    #n_samples = len(labels)
    #print("training")
    #param_grid = dict(C=2. ** np.arange(-4, 1))
    param_grid = dict(C=2. ** np.arange(-5, 4))
    #clf = LinearSVC(tol=1e-8, penalty='l2', dual=False, C=0.5)
    #clf = SVC(tol=1e-8, kernel='linear', probability=True)
    clf = LogisticRegression(tol=1e-8)

    #clf = NearestCentroid()

    #param_grid = dict(max_depth=np.arange(1, 10))
    #clf = RandomForestClassifier(n_estimators=10)

    grid = GridSearchCV(clf, cv=5, param_grid=param_grid, verbose=4)
    #print(cross_val_score(clf, counts, labels, cv=3))

    grid.fit(counts, labels)
    #clf.fit(X_train, y_train)
    #print(clf.score(X_test, y_test))
    comments_test, dates_test = load_test()
    counts_test = countvect.transform(comments_test)
    prob_pred = grid.best_estimator_.predict_proba(counts_test)
    tracer()
    write_test(prob_pred[:, 1])

if __name__ == "__main__":
    grid_search()
