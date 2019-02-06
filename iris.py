import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import pandas

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class Iris:
    def __init__(self):
        self.url = "iris.csv"
        self.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataset = pandas.read_csv(self.url, names=self.names)

        self.random_seed = 7

    def get_dimensions(self):
        return self.dataset.shape

    def peek(self, rows=20):
        return self.dataset.head(rows)

    def plot_univariates(self):
        self.dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        pyplot.show()

    def plot_histograms(self):
        self.dataset.hist()
        pyplot.show()

    def plot_scatter_matrix(self):
        scatter_matrix(self.dataset)
        pyplot.show()

    def split_training_set(self, validation_size=0.20):
        features = self.dataset.values[:, 0:4]
        label = self.dataset.values[:, 4]
        return model_selection.train_test_split(features, label, test_size=validation_size, random_state=self.random_seed)

    def get_algorithms(self):
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))
        return models

    def evaluate_algorithms(self):
        evaluation = []
        scoring = 'accuracy'
        x_train, x_validation, y_train, y_validation = self.split_training_set()

        for name, model in self.get_algorithms():
            kfold = model_selection.KFold(n_splits=10, random_state=self.random_seed)
            cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
            evaluation.append(cv_results)
            print ("Algorithm %s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

        return evaluation

    def plot_algorithm_comparison(self, results):
        algorithms = self.get_algorithms()

        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(algorithms)
        pyplot.show()


iris = Iris()

# print iris.get_dimensions()
# print iris.peek()
# print iris.dataset.describe()
# print iris.dataset.groupby('class').size()

#iris.plot_univariates()
#iris.plot_histograms()
#iris.plot_scatter_matrix()

evaluation = iris.evaluate_algorithms()
#iris.plot_algorithm_comparison(evaluation)

# Make predictions on validation dataset
x_train, x_validation, y_train, y_validation = iris.split_training_set()
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
