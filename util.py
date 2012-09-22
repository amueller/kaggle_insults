import numpy as np
from time import strftime
from IPython.core.debugger import Tracer

tracer = Tracer()


def preprocess_comment(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    #comment = comment.replace('.', ' ')
    comment = comment.replace("\\\\", "\\")
    return comment.decode('unicode-escape')


def deduplicate(comments, labels):
    hashes = np.array([hash(c) for c in comments])
    unique_hashes, indices = np.unique(hashes, return_inverse=True)
    doubles = np.where(np.bincount(indices) > 1)[0]
    mask = np.ones(len(comments), dtype=np.bool)
    # for each double entry
    for i in doubles:
        # mask out all but the first occurence
        not_the_first = np.where(indices == i)[0][1:]
        mask[not_the_first] = False
    return comments[mask], labels[mask]


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
    comments, labels = deduplicate(comments, labels)
    return comments, labels


def load_extended_data():
    comments, labels = load_data("train.csv")
    comments2, labels2 = load_data("test_with_solutions.csv")
    comments = np.hstack([comments, comments2])
    labels = np.hstack([labels, labels2])
    comments, labels = deduplicate(comments, labels)
    return comments, labels


def load_test(ds="test.csv"):
    print("loading test set")
    comments = []
    dates = []
    with open(ds) as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            dates.append(splitstring[0][:-1])
            comment = ",".join(splitstring[1:])
            comments.append(preprocess_comment(comment))
    comments = np.array(comments)
    return comments


def write_test(labels, fname=None, ds="test.csv"):
    if fname is None:
        fname = "test_prediction_september_%s.csv" % strftime("%d_%H_%M")
    with open(ds) as f:
        with open(fname, 'w') as fw:
            f.readline()
            fw.write("id,Insult,Date,Comment\n")
            for i, label, line in zip(np.arange(len(labels)), labels, f):
                fw.write("%d," % (i + 1))
                fw.write("%f," % label)
                fw.write(line)


def parse_subjectivity():
    strong_pos = []
    weak_pos = []
    weak_neg = []
    strong_neg = []
    with open("subjclueslen1-HLTEMNLP05.tff") as f:
        lines = f.readlines()
    for line in lines:
        # parse line, get rid of keys, only take values
        values = [c.split("=")[1] for c in line.strip().split(" ")]
        if values[5] == "negative":
            if values[0] == "weaksubj":
                weak_neg.append(values[2])
            else:
                strong_neg.append(values[2])
        elif values[5] == "positive":
            if values[0] == "weaksubj":
                weak_pos.append(values[2])
            else:
                strong_pos.append(values[2])
    lists = [strong_pos, strong_neg, weak_pos, weak_neg]
    lists = [np.unique(l) for l in lists]
    names = ["strong_pos", "strong_neg", "weak_pos", "weak_neg"]
    for n, l in zip(names, lists):
        with open(n + ".txt", "w") as f:
            f.writelines([w + "\n" for w in l])


def load_subjectivity():
    strong_pos = []
    weak_pos = []
    weak_neg = []
    strong_neg = []
    names = ["strong_pos", "strong_neg", "weak_pos", "weak_neg"]
    lists = [strong_pos, strong_neg, weak_pos, weak_neg]
    for n, l in zip(names, lists):
        with open(n + ".txt") as f:
            l.extend([w.strip() for w in f.readlines()])
    return [set(l) for l in lists]
