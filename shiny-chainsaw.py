from flask import Flask, render_template
from sklearn import datasets
from flask import request
from sklearn.tree import DecisionTreeClassifier
import pandas
import inspect
import json


app = Flask(__name__)
iris = datasets.load_iris()
X = pandas.DataFrame(iris.data)
y = pandas.DataFrame(iris.target)


def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

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
def train_tree():
    parameters = list(inspect.signature(DecisionTreeClassifier.__init__).
                      parameters.keys())
    parameters.remove('self')
    X_continuous_enabled, X_categorical_enabled, kwargs = [], [], {}
    for column_name in X.columns:
        if column_name in request.form:
            if request.form[column_name] == 'continuous':
                X_continuous_enabled.append(column_name)
            elif request.form[column_name] == 'categorical':
                X_categorical_enabled.append(column_name)
    for parameter in parameters:
        if parameter in request.form:
            kwargs[parameter] = request.form[parameter]
    X_select = pandas.DataFrame()
    for column_name in X_categorical_enabled:
        X_select.concat(pandas.get_dummies(X[column_name],
                                           prefix=column_name,
                                           prefix_sep="_"), axis=1)
    for column_name in X_continuous_enabled:
        X_select.concat(X[column_name], axis=1)
    classifier = DecisionTreeClassifier(**kwargs)
    if X_select.shape[0] == 0:
        X_select = X
    classifier.fit(X_select, y)
    with open('static/tree.json', 'w') as w:
        w.write(json.dumps(rules(classifier, features=X_select.columns, labels=['1', '2', '3'])))
    return render_template('index.html', parameters=parameters, column_names=X_select.columns)


if __name__ == '__main__':
    app.run()
