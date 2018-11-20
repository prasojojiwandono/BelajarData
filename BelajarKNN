##ini buat belajar KNN
## kodingan ini memerlukan file external (breast_cancer_data) (googling aja, mudah mendapatkannya)


import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import os

#import data
os.chdir('D:\data')
df = pd.read_csv('breast_cancer_data.txt')
list(df)


#mengelola missing data
df.replace('?',-999999,inplace = True)
df.drop(['id'],1,inplace=True)

#milih X dan Y
X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

#training
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)

accuracy = clf.score(X_test,Y_test)
print(accuracy)

##testing
example = np.array([4,6,3,1,4,4,8,7,3])
example = example.reshape(1,-1)
prediction = clf.predict(example)
print(prediction)
