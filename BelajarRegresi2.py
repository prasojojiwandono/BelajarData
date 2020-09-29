#ini kodingan tentang belajar regresi

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
style.use('ggplot')
import random

#ploting data
xs = np.array([1,2,3,4,5,6],dtype = np.float64)
ys = np.array([5,4,6,5,6,7],dtype = np.float64)
plt.scatter(xs,ys)
plt.show()

##rumus regresi
def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs)*mean(ys))-mean(xs*ys))/
         ((mean(xs)*mean(xs))-mean(xs*xs)))
    b = mean(ys) - m * (mean(xs))
    return m,b

##ploting garis regresi
m,b = best_fit_slope_and_intercept(xs,ys)
garis_regresi = [(m*x)+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs,garis_regresi)
plt.show()

##rumus r-squared
def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

r_squared = coefficient_of_determination(ys,garis_regresi)

print(r_squared)



##coba buat data baru
def create_dataset(hm,variance,step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val += step
        elif correlation and correlation =='neg':
            val -= step
    
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype = np.float64),np.array(ys,dtype = np.float64)
    
    
##testing
xa,ya = create_dataset(40,10,2,correlation='pos')
ma,ba = best_fit_slope_and_intercept(xa,ya)
garis_regresi_a = [(ma*x)+ba for x in xa]
r_square_a = coefficient_of_determination(ya,garis_regresi_a)
print(r_square_a,ma,ba)
plt.scatter(xa,ya)
plt.plot(xa,garis_regresi_a)
plt.show()
