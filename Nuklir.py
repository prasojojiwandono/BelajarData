##ini adalah kodingan tentang prediksi sensor nuklir
## di kodingan ini perlu file eksternal


import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import train_test_split
from statistics import mean
import os
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle


##import data
os.chdir('D:\data\pickle')
dfjun = pd.read_pickle("./sensorjuni.pkl")
dfjan = pd.read_pickle("./sensorjanuari.pkl")
os.chdir('D:\data') 

#deskripsi data
dfjun.describe()


#sampling data
samp = np.absolute(np.arange(0,len(dfjun),10))
traindata = dfjun.iloc[samp,1:28]

a=np.arange(27)
a = np.delete(a,[22])#sensor nomor 22 ga usah dipake
nptraindata = np.array(traindata)
nptraindata = nptraindata[:,a]


##melihat plot dari sensor suhu
suhu = nptraindata[:,8]
plt.plot(suhu)
plt.show()


##mensorting data lagi agar training bisa lebih mudah (pensortingan dilakukan atas dasar pengamatan sensor suhu)
awal = nptraindata[0:7000:,]
akhir = nptraindata[len(nptraindata)-7000:len(nptraindata):,]
tengah = nptraindata[7000:len(nptraindata)-7000:10,:]
nptraindata = np.concatenate((awal,tengah,akhir),axis=0)

#pembagian training vs testing
X_train,X_test,y_train,y_test = train_test_split(nptraindata,nptraindata,test_size=0.2)

#normalisasi
mmsc = MinMaxScaler()
Xanorm = mmsc.fit_transform(X_train)
Yanorm = mmsc.fit_transform(y_train)
Xenorm = mmsc.fit_transform(X_test)
Yenorm = mmsc.fit_transform(y_test)


#training
model = Sequential()
model.add(Dense(15,input_dim = 26,activation='relu'))
model.add(Dense(26,activation='sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(Xanorm,Yanorm,epochs = 300,batch_size= 100)


#evaluasi
nilaitesting = model.evaluate(Xenorm,Yenorm)


#ploting prediksi vs asli
Panorm = model.predict(Xenorm)
asli = Xenorm[:,7]
prediksi = Panorm[:,7]
plt.plot(asli,label = 'asli')
plt.plot(prediksi,label = 'prediksi')
plt.legend(loc='best')
plt.show()


#ploting prediksi vs asli lagi
nomorsensor = 7
awal = nptraindata[0:19800,:]
mmsc = MinMaxScaler()
anorm = mmsc.fit_transform(awal)
# anorm = (awal-mi)/(ma-mi)
s7 = anorm[:,nomorsensor]
prediksi = model.predict(anorm)
prediksis7 = prediksi[:,nomorsensor]
plt.plot(s7,label = 'asli')
plt.plot(prediksis7,label = 'prediksi')
plt.legend(loc='best')
plt.show()


##mempersiapkan data testing yang baru
awalan = dfjan.iloc[:,2]
npmentah = np.array(awalan)
npawalan = npmentah[0:172800]
npawalan = npawalan.reshape(len(npawalan),1)
for i in np.arange(1,27):
    bb = npmentah[i*172800:(i+1)*172800]
    bb = bb.reshape(len(bb),1)
    npawalan = np.concatenate((npawalan,bb),axis=1)

a=np.arange(27)
a = np.delete(a,[22])#sensor nomor 22 ga usah dipake
nptestdata = npawalan.copy()
nptestdata = nptestdata[:,a]


#sampling testing data
samptest = nptestdata[0:len(nptestdata):10]


#ploting asli vs prediksi
nomorsensor = 7
mmsc = MinMaxScaler()
testnorm = mmsc.fit_transform(samptest)
s7 = testnorm[:,nomorsensor]
prediksi = model.predict(testnorm)
prediksis7 = prediksi[:,nomorsensor]
plt.plot(s7,label = 'asli')
plt.plot(prediksis7,label = 'prediksi')
plt.legend(loc='best')
plt.show()

