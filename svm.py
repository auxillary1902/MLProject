import pandas
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
dataframe = pandas.read_csv("satisfactionmain.csv")
array = dataframe.values
# X1 = dataframe.iloc[:,7:23]
# Y1 = dataframe.iloc[:,23:24]
# print(Y1)
print("Read the dataset. Splitting as training attributes and target attributes")
X = array[:100000,7:23]
Y = array[:100000,23]
print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3,random_state=109) # 70% training and 30% test
print("The training set is after 70 percent handout:")

print(X_train)
print(y_train)
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))

# # Model Recall: what percentage of positive tuples are labelled as such?
# print("Recall:",metrics.recall_score(y_test, y_pred))