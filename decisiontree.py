import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.externals.six import StringIO  
import pydot 
import csv
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv( "satisfactionmain.csv") 
      
    # Printing the dataswet shape 
    #print ("Dataset Lenght: ", len(balance_data)) 
    #print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    #print ("Dataset: ",balance_data.head()) 
    return balance_data 
  
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 7:23] 
    Y = balance_data.values[:, 23] 
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
    print("The training data set after 70 percent handout is :")
    print(X_train)
    print("The target training attribute:")
    print(y_train)
    print("Number of training records: " + str(len(X_train)))
    print("Number of test records: " + str(len(X_test)))
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = tree.DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    dot_data = StringIO() 
    tree.export_graphviz(clf_gini, out_file=dot_data,class_names=clf_gini.classes_) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph[0].write_pdf("ginigraph.pdf") 
    print("Decision tree has been printed as a file for gini index")
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = tree.DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 5, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    dot_data = StringIO() 
    tree.export_graphviz(clf_entropy, out_file=dot_data,class_names=clf_entropy.classes_) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph[0].write_pdf("entropygraph.pdf") 
    print("Decision tree has been printed as a file for entropy")

    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    print("The precision metrics have been printed as a report")

def classifaction_report_csv(report,filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = (row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)

  
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
      
    print("Gini results precision reports:")
    print(classification_report(y_test, y_pred_gini))
    print("Entropy results precision reports:")
    print(classification_report(y_test,y_pred_entropy))
      
# Calling main function 
if __name__=="__main__": 
    main() 