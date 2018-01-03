import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

Exotrain=pd.read_csv("ExoTrain.csv")
FinalTest=pd.read_csv("FinalTest.csv")

import numpy as np

y=Exotrain['LABEL']
x=Exotrain.drop(labels='LABEL',axis=1)
clf=MLPClassifier(solver='lbfgs',alpha=0.1,hidden_layer_sizes=(10,5),random_state=1)

from sklearn import preprocessing

x_scale=preprocessing.scale(x)
clf.fit(x_scale,y)
x_test=FinalTest.drop(labels='LABEL',axis=1)
y_test=FinalTest['LABEL']
y_pred=clf.predict(x_test)
y_pred1=clf.predict(x)
print(y_pred)
#print(metrics.accuracy_score(y_pred,y_test))
print("on training data")
print(metrics.accuracy_score(y_pred1,y))
print(y_pred1)
#for i in y_pred :
    #print (i)

    


