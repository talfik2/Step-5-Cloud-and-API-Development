
# Save Model Using Pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle as pkl
import joblib

first3 = "C:\\Users\\talfi\\python\\dataglacier\\w4\\deneme\\first3.csv"
#Y = "C:\\Users\\talfi\\python\\dataglacier\\w4\\deneme\\y.csv"
#include = ['City','Age','Population']#population'a dikkat et
dataframe = pd.read_csv(first3)
y = pd.read_csv(Y)
#dataframe_ = dataframe(include)
X_train, X_test, y_train ,y_test = train_test_split(
    dataframe.drop(columns = "Company"),
    dataframe["Company"],
    test_size=0.25,
    random_state=42,
    stratify=y
)
dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=16, min_samples_split=8,
                       random_state=42)
# Fit the model on training set
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)


# save the model to disk with joblib
filename = 'model.pkl'
joblib.dump(dtc, 'model.pkl')
# save the model to disk with pickle
#pickle.dump(dtc, open(filename, 'wb'))
 
# some time later...

#load the model from disk with joblib
dtc = joblib.load('model.pkl')
# load the model from disk with pickle
#dtc.pkl = pickle.load(open(filename, 'rb'))
#result = dtc.score(X_test,y_test)
#print(result)