# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

dataframe = pandas.read_csv("satisfactionmain.csv")
array = dataframe.values
# X1 = dataframe.iloc[:,7:23]
# Y1 = dataframe.iloc[:,23:24]
# print(Y1)
print("Read the dataset. Splitting as training attributes and target attributes")
X = array[:,7:23]
Y = array[:,23]
# print(Y)
print(X)
print(Y)
seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
gnb = GaussianNB()

num_trees = 6
model = BaggingClassifier(base_estimator=gnb, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Performing bagging with 10-fold cross-validation. The accuracy for each of the 10 folds are:")
print(results)
print("Final Accuracy : " + str(results.mean() * 100) + "%")
