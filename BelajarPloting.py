#ini tentang belajar ploting

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os



####ses 1
##x = [1,2,3,4,5,6,7]
##y = [50,51,52,48,47,49,46]

#plt.plot(x,y,'rD')  #coba aja
#plt.plot(x,y,'D')  #coba aja
#plt.plot(x,y,'D--')
##plt.plot(x,y,color = 'green',marker = '.',linestyle = '-.')
##plt.show()



####ses 2

##days = [1,2,3,4,5,6,7]
##max_t = [50,51,52,48,47,49,46]
##min_t = [43,42,40,44,33,35,37]
##avg_t = [45,48,48,46,40,42,41]
##
##plt.xlabel('days')
##plt.ylabel('temperature')
##plt.title('weather')
##plt.plot(days,max_t,label = 'max')
##plt.plot(days,min_t,label = 'min')
##plt.plot(days,avg_t,label = 'avg')
##plt.legend(loc = 'best')
##plt.grid()
##plt.show()

##ses 3 ##bar chart
##company = ['gogel','amajon','mikocok','pesbuk']
##revenue =[90,136,89,27]
##profit = [40,2,34,12]
##
##xpos = np.arange(len(company))
##plt.xticks(xpos,company)
####plt.bar(company,revenue,label='revenue')
####plt.bar(company,profit,label='profit')
####plt.bar(xpos+0.2,revenue,width=0.4,label='revenue')
####plt.bar(xpos-0.2,profit,width=0.4,label='profit')
##plt.barh(xpos+0.2,revenue,label='revenue')
##plt.barh(xpos-0.2,profit,label='profit')
##plt.xlabel('revenue')
##plt.ylabel('company')
##plt.legend()
##plt.show()


##ses 4 ## histogram
##aaa =[5.1,	4.9,	4.7,	4.6,	5.0,	5.4,	4.6,	5.0,	4.4,	4.9,	5.4,	4.8,	4.8,	4.3,	5.8,	5.7,	5.4,	5.1,	5.7,	5.1,	5.4,	5.1,	4.6,	5.1,	4.8,	5.0,	5.0,	5.2,	5.2,	4.7,	4.8,	5.4,	5.2,	5.5,	4.9,	5.0,	5.5,	4.9,	4.4,	5.1,	5.0,	4.5,	4.4,	5.0,	5.1,	4.8,	5.1,	4.6,	5.3,	5.0,	7.0,	6.4,	6.9,	5.5,	6.5,	5.7,	6.3,	4.9,	6.6,	5.2,	5.0,	5.9,	6.0,	6.1,	5.6,	6.7,	5.6,	5.8,	6.2,	5.6,	5.9,	6.1,	6.3,	6.1,	6.4,	6.6,	6.8,	6.7,	6.0,	5.7,	5.5,	5.5,	5.8,	6.0,	5.4,	6.0,	6.7,	6.3,	5.6,	5.5,	5.5,	6.1,	5.8,	5.0,	5.6,	5.7,	5.7,	6.2,	5.1,	5.7,	6.3,	5.8,	7.1,	6.3,	6.5,	7.6,	4.9,	7.3,	6.7,	7.2,	6.5,	6.4,	6.8,	5.7,	5.8,	6.4,	6.5,	7.7,	7.7,	6.0,	6.9,	5.6,	7.7,	6.3,	6.7,	7.2,	6.2,	6.1,	6.4,	7.2,	7.4,	7.9,	6.4,	6.3,	6.1,	7.7,	6.3,	6.4,	6.0,	6.9,	6.7,	6.9,	5.8,	6.8,	6.7,	6.7,	6.3,	6.5,	6.2,	5.9]
##bbb = [3.5,	3.0,	3.2,	3.1,	3.6,	3.9,	3.4,	3.4,	2.9,	3.1,	3.7,	3.4,	3.0,	3.0,	4.0,	4.4,	3.9,	3.5,	3.8,	3.8,	3.4,	3.7,	3.6,	3.3,	3.4,	3.0,	3.4,	3.5,	3.4,	3.2,	3.1,	3.4,	4.1,	4.2,	3.1,	3.2,	3.5,	3.1,	3.0,	3.4,	3.5,	2.3,	3.2,	3.5,	3.8,	3.0,	3.8,	3.2,	3.7,	3.3,	3.2,	3.2,	3.1,	2.3,	2.8,	2.8,	3.3,	2.4,	2.9,	2.7,	2.0,	3.0,	2.2,	2.9,	2.9,	3.1,	3.0,	2.7,	2.2,	2.5,	3.2,	2.8,	2.5,	2.8,	2.9,	3.0,	2.8,	3.0,	2.9,	2.6,	2.4,	2.4,	2.7,	2.7,	3.0,	3.4,	3.1,	2.3,	3.0,	2.5,	2.6,	3.0,	2.6,	2.3,	2.7,	3.0,	2.9,	2.9,	2.5,	2.8,	3.3,	2.7,	3.0,	2.9,	3.0,	3.0,	2.5,	2.9,	2.5,	3.6,	3.2,	2.7,	3.0,	2.5,	2.8,	3.2,	3.0,	3.8,	2.6,	2.2,	3.2,	2.8,	2.8,	2.7,	3.3,	3.2,	2.8,	3.0,	2.8,	3.0,	2.8,	3.8,	2.8,	2.8,	2.6,	3.0,	3.4,	3.1,	3.0,	3.1,	3.1,	3.1,	2.7,	3.2,	3.3,	3.0,	2.5,	3.0,	3.4,	3.0]
##plt.hist([aaa,bbb],rwidth=0.95,color =['g','r'],label=['aaa','bbb'],orientation='horizontal')
##plt.legend()
##plt.show()

## ses 5
vals =[1000,600,300,410,250]
tipe = ['tipe A','tipe B','tipe C','tipe D','tipe E']
plt.pie(vals,labels=tipe,autopct='%0.1f%%',explode=[0,0.1,0,0,0])
plt.axis('equal')
plt.savefig('piechart.png')
plt.show()







