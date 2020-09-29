import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

##s = pd.Series([1,3,5,np.nan,6,8])
##print(s)

dates = pd.date_range('20130101',periods=6)
##print(dates)

#df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
#print(df)


####################################################################

##df2 = pd.DataFrame({
##    'A':1.,
##    'B':pd.Timestamp('20130102'),
##    'C':pd.Series(1,index=list(range(4)),dtype='float32'),
##    'D':np.array([3]*4,dtype = 'int32'),
##    'E':pd.Categorical(["test","train","test","train"]),
##    'F':'foo'
##    })
#print(df2)
#print(df2.dtypes)


#######################################################################

#df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df.head())
##print(df.tail(3))
##print(df.index)
##print(df.columns)
##print(df.values)
##print(df.describe())
#print(df.sort_index(axis=1,ascending=False))
#print(df.sort_values(by='B'))


##df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df['A'])
##print(df[0:3])
##print(df['20130101':'20130104'])




###################################################
##df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df.loc[dates[0]])
##print(df.loc[:,['A','B']])
##print(df.loc['20130102':'20130105',['A','B']])



#############################################
##df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df.loc[dates[0]])
##print(df.loc[:,['A','B']])
##print(df.loc['20130102':'20130105',['A','B']])

#############################################
##df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df.iloc[[0,3]])
##AA=[2,4]
##print(df.iloc[AA])
##print(df.iloc[3:5,0:2])
##print(df.iloc[1,2])

##df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
##print(df['B'][df.A>0])
##df2 = df.copy()
##df2['B'][df2['A']>0]=3
##print(df2)
##print(df)


df = pd.DataFrame(np.random.randn(6,4),index=dates,columns = list('ABCD'))
df3 = df.copy()
df3['E'] = ['one','one','two','three','four','three']
print(df3)
print(df3[df3['E'].isin(['two','four'])])



print('hello world')
