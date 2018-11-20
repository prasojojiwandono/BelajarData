##ini kodingan tentang analisa saham
##dikodingan ini memerlukan data eksternal yang bisa didownload via yahoo finance


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


#ambil data dari D:\data (local directory)
os.chdir('D:\data')
df = pd.read_csv('xlcsv2.csv')

##sorting data yang hanya diperlukan
df=df.dropna()
df = df[df.Volume > 1 ]
df = df[::-1]
npxl=np.array(df.drop(['Date','Adj Close'],1))
npxl.dtype = np.float64


##rumus - rumus

def gradienregresi(close,jumlahbaris=30):
    b = np.arange(1,jumlahbaris + 1)
    b = b[::-1]
    #b.dtype = np.float64
    c = np.array(b,dtype = np.float64)
    jml = len(close)-jumlahbaris + 1
    npjml = np.arange(jml)
    npjml = np.array(npjml,dtype = np.float64)
    for i in np.arange(jml):
        coba = close[i:jumlahbaris+i]
        coba = coba.reshape(len(coba),)
        m = ( ((mean(c)*mean(coba))-mean(c*coba))/
         ((mean(c)*mean(c))-mean(c*c)))/mean(coba)
        npjml[i]= m 
    return npjml
	
	
def gradma(close,jumlahbaris):
    jml = len(close)-jumlahbaris + 1
    npjml = np.arange(jml)
    npjml = np.array(npjml,dtype = np.float64)
    for i in np.arange(jml):
        sblm = close[i+1:jumlahbaris+i+1]
        sblm = sblm.reshape(len(sblm),)
        ssdh = close[i:jumlahbaris+i]
        ssdh = ssdh.reshape(len(ssdh),)
        m = (mean(ssdh)- mean(sblm))/mean(sblm)
        npjml[i]= m 
    return npjml


	
	
def gradclose(close):
    jml = len(close)- 1
    npjml = np.arange(jml)
    npjml = np.array(npjml,dtype = np.float64)
    for i in np.arange(jml):
        sblm = close[i+1:i+2]
        sblm = sblm.reshape(len(sblm),)
        ssdh = close[i:i+1]
        ssdh = ssdh.reshape(len(ssdh),)
        m = (mean(ssdh)- mean(sblm))/mean(sblm)
        npjml[i]= m 
    return npjml
	
	
def getrsi(close,jumlahbaris=14):
    csblm = close[1:]
    cssdh = close[:-1]
    cselisih = cssdh - csblm
    cup = np.copy(cselisih)
    cdown = np.copy(cselisih)
    cup[cup<0]=0
    cdown[cdown>0]=0
    cdown = cdown * -1
    cup = cup.reshape(len(cup),)
    cdown = cdown.reshape(len(cdown),)
    jml = len(close)-jumlahbaris -1
    npjml = np.arange(jml)
    npjml = np.array(npjml,dtype = np.float64)
    for i in np.arange(jml):
        u = mean(cup[i:jumlahbaris+i])
        d = mean(cdown[i:jumlahbaris+i])
        banding = u/d
        rsi = 100 - (100/(banding+1))
        npjml[i]= rsi
    return npjml	
	
	
def getpersenvolum(vol,jumlahbaris = 5):
    jml = len(vol)-jumlahbaris + 2
    npjml = np.arange(jml)
    npjml = np.array(npjml,dtype = np.float64)
    for i in np.arange(jml):
        bb = vol[i:jumlahbaris+i]
        bb = bb.reshape(len(bb),)
        mbb = mean(bb)
        uu = (vol[i]-mbb)*100/mbb
        npjml[i] = uu
    return npjml
	
	
def kalilipat(vol):
    vsblm = vol[1:]
    vssdh = vol[:-1]
    kk = vssdh/vsblm
    return kk
	

def phigh(high,close):
    gg = (high-close)/close
    return gg
	
	
def plow(low,close):
    nn = (close-low)/close
    return nn
	
	
def tabel2(npxl,preg= 30,pma5=5,pma10=10,prsi = 14,ppv = 5,geser = 1):
    a = npxl.shape[1]
    volum = npxl[:,a-1:a]
    close = npxl[:,a-2:a-1]
    low = npxl[:,a-3:a-2]
    high = npxl[:,a-4:a-3]
    open = npxl[:,a-5:a-4]
    csblm=close[geser:]
    cssdh = close[:-geser]
    cselisih = cssdh - csblm
    cnt = cselisih
    cnt[cnt > 0]= 1
    cnt[cnt<=0] = -1
    reg = gradienregresi(close,preg)
    ma5 = gradma(close,pma5)
    ma10 = gradma(close,pma10)
    gc = gradclose(close)
    rsi = getrsi(close,prsi)
    pv = getpersenvolum(volum,ppv)
    kl = kalilipat(volum)
    ph = phigh(high,close)
    pl = plow(low,close)
    cl = np.copy(close)
    #############################kalo mau nambah
    reg = reg.reshape(len(reg),1)
    ma5 = ma5.reshape(len(ma5),1)
    ma10 = ma10.reshape(len(ma10),1)
    gc = gc.reshape(len(gc),1)
    rsi = rsi.reshape(len(rsi),1)
    pv = pv.reshape(len(pv),1)
    kl = kl.reshape(len(kl),1)
    ph = ph.reshape(len(ph),1)
    pl = pl.reshape(len(pl),1)
    cnt = cnt.reshape(len(cnt),1)
    cl = cl.reshape(len(cl),1)
    
    ####################kalo mau nambah
    hh = np.array([reg.shape[0],ma5.shape[0],ma10.shape[0],gc.shape[0],rsi.shape[0],pv.shape[0],kl.shape[0],ph.shape[0],pl.shape[0],cnt.shape[0],cl.shape[0]])
    ##kalo ada penambahan hh juga ditambahin
    mi = np.min(hh)
    if mi == len(reg):
        reg = reg[:]
    else:
        reg = reg[:-(len(reg)-mi)]
    
    if mi == len(reg):
        reg = reg[:]
    else:
        reg = reg[:-(len(reg)-mi)]
    
    if mi == len(ma5):
        ma5 = ma5[:]
    else:
        ma5 = ma5[:-(len(ma5)-mi)]
    
    if mi == len(ma10):
        ma10 = ma10[:]
    else:
        ma10 = ma10[:-(len(ma10)-mi)]
        
    if mi == len(gc):
        gc = gc[:]
    else:
        gc = gc[:-(len(gc)-mi)]
    
    if mi == len(rsi):
        rsi = rsi[:]
    else:
        rsi = rsi[:-(len(rsi)-mi)]
        
    if mi == len(pv):
        pv = pv[:]
    else:
        pv = pv[:-(len(pv)-mi)]
        
    if mi == len(kl):
        kl = kl[:]
    else:
        kl = kl[:-(len(kl)-mi)]
      
    if mi == len(ph):
        ph = ph[:]
    else:
        ph = ph[:-(len(ph)-mi)]
    
    if mi == len(pl):
        pl = pl[:]
    else:
        pl = pl[:-(len(pl)-mi)]
        
    if mi == len(cnt):
        cnt = cnt[:]
    else:
        cnt = cnt[:-(len(cnt)-mi)]
        
    if mi == len(cl):
        cl = cl[:]
    else:
        cl = cl[:-(len(cl)-mi)]
    #####perlu juga
    
    
    
    ##ditabelnya juga perlu ditambahin
    tabel1 = np.concatenate((reg,ma5,ma10,gc,rsi,pv,kl,ph,pl,cl),axis=1)
    tabel1 = tabel1[geser:]
    #cnt = cnt[:-1]
    cl = cl[:-geser]
    tabelnya = np.concatenate((tabel1,cl),axis=1)
    return tabelnya
    
 
sxl = tabel2(npxl)
X=sxl[:,:-1]
Y=sxl[:,sxl.shape[1]-1:]

#membagi training dan testing
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.2)

#normalisasi
scx = MinMaxScaler().fit(Xtrain)
scy = MinMaxScaler().fit(Ytrain)
Xnorm = scx.transform(Xtrain)
Ynorm = scy.transform(Ytrain)


#training
model = Sequential()
model.add(Dense(20,activation='relu',input_dim=Xnorm.shape[1]))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(Xnorm,Ynorm,epochs = 200,batch_size = 100)

#evaluasi
score = model.evaluate(Xenorm,Yenorm)
print(score[0])

##melihat prediksi vs asli
prediksi = model.predict(Xnorm)
plt.plot(prediksi[:30])
plt.plot(Ynorm[:30])
plt.show()



##melihat asli vs prediksi lagi
Xrnorm = scx.transform(X)
Yrnorm = scy.transform(Y)
prediksi3 = model.predict(Xrnorm,batch_size = 100)
prediksi3=prediksi3[::-1]
Yrnorm=Yrnorm[::-1]
Xrnorm=Xrnorm[::-1]
plt.plot(prediksi3[len(prediksi3)-30:],label='prediksi')
plt.plot(Yrnorm[len(Yrnorm)-30:],label='asli')
plt.legend()
plt.show()

