from flask import Flask
from sklearn import datasets
from flask import request
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas
import inspect


app = Flask(__name__)
iris = datasets.load_iris()
X = pandas.DataFrame(iris.data)
y = pandas.DataFrame(iris.target)


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
    classifier.fit(X_select, y)
    export_graphviz(classifier, out_file='static/tree.dot')
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
