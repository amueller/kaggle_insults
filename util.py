import numpy as np
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
            # the remaining commata where in the text, replace them
            comment = ",".join(splitstring[2:])
            comment = comment.strip().strip('"')
            comment = comment.replace('_', ' ')
            comment = comment.replace('.', ' ')
            comment = comment.replace("\\\\", "\\")
            comment = comment.decode('unicode-escape')
            comments.append(comment)
    labels = np.array(labels, dtype=np.int)
    dates = np.array(dates)
    comments = np.array(comments)
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
            comment = comment.replace('_', ' ')
            comment = comment.replace('.', ' ')
            comment = comment.replace("\\\\", "\\")
            comment = comment.decode('unicode-escape')
            comments.append(comment)
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
