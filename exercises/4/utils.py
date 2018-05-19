#!/usr/bin/env python

from os import chmod

import numpy as np
import matplotlib.pyplot as plt
import sys


def output(name, s):
    print s
    path = '{0}.result.txt'.format(name)
    f = open(path, 'w')
    f.write(str(s))
    f.close()

    if 'linux' in sys.platform:
        chmod(path, 0o777)


def plt_save(name, title=None):
    if title:
        plt.title(title)
    path = '{0}.png'.format(name)
    plt.savefig(path, format='png')
    if 'linux' in sys.platform:
        chmod(path, 0o777)

    plt.close('all')


def plot(name='plot', title=None, xlabel='x', ylabel='y', yl=None, yh=None, ytick=None, xl=None, xh=None, xtick=None, grid=True, legend_loc='best'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)

    if yl and yh:
        if ytick:
            plt.gca().set_yticks(np.arange(yl, yh + ytick, ytick))
        else:
            plt.gca().set_ylim((yl, yh))
    
    if xl and xh:
        if xtick:
            plt.gca().set_xticks(np.arange(xl, xh + xtick, xtick))
        else:
            plt.gca().set_xlim((xl, xh))

    if grid:
        plt.grid()

    plt_save(name, title)


def image(vector, width=28, height=28):
    # plt.imshow(np.reshape(vector, (height, width)), interpolation='nearest', cmap=plt.cm.gray_r)
    fig = plt.imshow(np.reshape(vector, (height, width)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def images(vectors, titles, n_cols, width=28, height=28):
    n_rows = len(vectors) / n_cols if len(vectors) % n_cols == 0 else (len(vectors) / n_cols) + 1
    for i, (vector, title) in enumerate(zip(vectors, titles)):
        plt.subplot(n_rows, n_cols, i+1)
        plt.title(title)
        image(vector.astype(float), height, width)

    plt.tight_layout()


def table(row_labels, col_labels, rows):
    _, ax = plt.subplots()
    ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)

    plt.table(
        rowLabels=row_labels,
        colLabels=col_labels,
        cellText=rows,
        loc='center'
    )

    ax.axis('off')


def mark(xy, xytext=None, text=None):
    x, y = xy
    plt.plot(x, y, 'g*')
    if xytext and text:
        plt.annotate(
            text,
            xy=xy, 
            xytext=xytext,
            arrowprops={'facecolor': 'black', 'shrink': 0.05}
        )


def calc_accuracy_of_predictions(predictions, labels):
    correct = sum(p == labels[i] for i, p in enumerate(predictions))
    return float(correct) / len(predictions)


# the score of predictor (algorithm accuracy)
def score(predict_func, data, labels):
    return calc_accuracy_of_predictions(predict_func(data), labels)


def norm(vector):
    return np.linalg.norm(vector, ord=2)


def normalize(vector):
    norm = np.linalg.norm(vector, ord=2)
    if norm == 0:
        return vector
    return vector / norm


def main(argv, *subquestions):
    functions = {s.__name__: s for s in subquestions}

    if '--help' in argv or len(argv) == 1:
        print 'Use subquestion as flags in order to get the relevant plots, or use -A/--all flag to get them all.'
        return 
    
    if '-A' in argv or '--all' in argv:
        for f in functions.values():
            f()
        return
    
    for subquestion in argv[1:]:
        if subquestion[1:] in functions:
            functions[subquestion[1:]]()

