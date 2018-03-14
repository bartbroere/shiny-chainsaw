import inspect
import json
import re

import pandas
from flask import Flask, render_template
from flask import request
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
iris = datasets.load_iris()
X = pandas.DataFrame(iris.data, columns=iris.feature_names)
y = pandas.DataFrame(iris.target)


def rules(clf, features, labels, node_index=0):
    """
    Represents nodes in a decision tree and their children nodes as
    nested dictionaries

    :param clf: an instance of an sklearn decision tree classifier
    :param features: names of the features in the classifier
    :param labels: names of the labels in the classifier
    :param node_index: index of node used for recursion over nodes
    :return: a node with its subnodes
    :return: the full decision tree if node_index == 0
    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node


@app.route('/', methods=['POST', 'GET'])
def load_data_dialog():
    """

    :return:
    """


@app.route('/train', methods=['POST', 'GET'])
def train_tree():
    """
    Re-train a decision tree based on the input of a web page.

    :return: A new version of the webpage, with a re-trained decision tree.
    """
    parameters = list(inspect.signature(DecisionTreeClassifier.__init__).
                      parameters.items())
    parameters.pop(0)
    X_continuous_enabled, X_categorical_enabled, kwargs = [], [], {}
    for column_name in X.columns:
        if column_name in request.args:
            if request.args[column_name] == 'continuous':
                X_continuous_enabled.append(column_name)
            elif request.args[column_name] == 'categorical':
                X_categorical_enabled.append(column_name)
    for parameter in parameters:
        parameter = parameter[0]
        if parameter in request.args:
            if request.args[parameter] is not '':
                floats = re.findall("\d+\.\d+", request.args[parameter])
                ints = re.findall("^\d+$", request.args[parameter])
                if len(floats) == 1:
                    kwargs[parameter] = float(request.args[parameter])
                elif len(ints) == 1:
                    kwargs[parameter] = int(request.args[parameter])
                elif request.args[parameter] == "None":
                    kwargs[parameter] = None
                elif request.args[parameter] == "False":
                    kwargs[parameter] = False
                elif request.args[parameter] == "True":
                    kwargs[parameter] = True
                else:
                    kwargs[parameter] = request.args[parameter]
    X_select = pandas.DataFrame()
    for column_name in X_categorical_enabled:
        X_select = pandas.concat([X_select,
                                  pandas.get_dummies(X[column_name],
                                                     prefix=column_name,
                                                     prefix_sep="_")],
                                 axis=1)
    for column_name in X_continuous_enabled:
        X_select = pandas.concat([X_select, X[column_name]], axis=1)
    print(X_select.shape)
    classifier = DecisionTreeClassifier(**kwargs)
    if X_select.shape[0] == 0:
        X_select = X
    classifier.fit(X_select, y)
    with open('static/tree.json', 'w') as w:
        w.write(json.dumps(rules(classifier,
                                 features=X_select.columns,
                                 labels=list(y[0].unique()))))
    parameters = [(parameter_name,
                   parameter.default,
                   kwargs.get(parameter_name, ''))
                  for parameter_name, parameter in parameters]
    return render_template('index.html',
                           parameters=parameters,
                           column_names=X.columns)


if __name__ == '__main__':
    app.run()
