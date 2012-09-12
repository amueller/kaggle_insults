import numpy as np
from time import strftime
from IPython.core.debugger import Tracer

tracer = Tracer()


def preprocess_comment(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    comment = comment.replace('.', ' ')
    comment = comment.replace("\\\\", "\\")
    return comment.decode('unicode-escape')


def load_data(ds="train.csv"):
    print("loading")
    comments = []
    dates = []
    labels = []
    with open(ds) as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            labels.append(splitstring[0])
            dates.append(splitstring[1][:-1])
            # the remaining commata where in the text, replace them
            comment = ",".join(splitstring[2:])
            comments.append(preprocess_comment(comment))
    labels = np.array(labels, dtype=np.int)
    dates = np.array(dates)
    comments = np.array(comments)
    return comments, labels


def load_extended_data():
    comments, labels = load_data("train.csv")
    comments2, labels2 = load_data("test_with_solutions.csv")
    comments = np.hstack([comments, comments2])
    labels = np.hstack([labels, labels2])
    return comments, labels


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
            comments.append(preprocess_comment(comment))
    dates = np.array(dates)
    comments = np.array(comments)
    return comments, dates


def write_test(labels, fname=None):
    if fname is None:
        fname = "test_prediction_september_%s.csv" % strftime("%d_%H_%M")
    with open("test.csv") as f:
        with open(fname, 'w') as fw:
            f.readline()
            fw.write("Insult,Date,Comment\n")
            for label, line in zip(labels, f):
                fw.write("%f," % label)
                fw.write(line)
