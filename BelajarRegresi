## ini kodingan tentang belajar regresi
## di dalam kodingan ini membutuhukan file tambahan dari luar (file csv) yang bisa didapatkan (didownload) dari situs yahoo finance (https://finance.yahoo.com/quote/EXCL.JK/history?p=EXCL.JK)

import os
import pandas as pd
import numpy as np
import math,datetime
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import pickle


## mengganti gaya plot ## ini optional
style.use('ggplot')


## setelah data di download dari internet, saya memasukankan ke dalam folder D:\data , lalu saya namai filenya dengan nama EXCL.csv
os.chdir('D:\data')
df=pd.read_csv('EXCL.csv')

## melihat kolom apa saja yang ada dalam data tersebut
list(df)

## memilih kolom tertentu yang hanya diperlukan 
df = df[['Date','Open','High','Low','Close','Volume']]

##hanya memilih data dengan volumenya ada, (didasarkan pada pengetahuan saham) 
df = df[df.Volume > 1]

##mengantisipasi missing data
df.fillna(-99999,inplace=True)

## membuat perhitungan
df['HL_PCT']=(df['High']-df['Close'])/df['Close']*100.0
df['PCT_CHANGE']=(df['Close']-df['Open'])/df['Open']*100.0

## memilih kolom tertentu yang hanya diperlukan
df=df[['Close','HL_PCT','PCT_CHANGE','Volume']]

##membentuk kolom target (disini namanya label)
forecast_col = 'Close'
forecast_out = int(math.ceil(0.005*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

## memilih parameter X dan Y
X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])

##membagi testing dan training
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

##mentraining
clf = LinearRegression()
clf.fit(X_train,Y_train)

##melihat score testing
clf.score(X_test,Y_test)



