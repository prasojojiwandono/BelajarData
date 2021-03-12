import mplfinance as mpf
import pandas as pd
from matplotlib.pyplot import savefig
import os


# directory = '/home/prasojo/belajar/BelajarData/dataset/saham/emiten'
# pic_directory = '/home/prasojo/belajar/BelajarData/dataset/saham/emiten_pic'

def iterate_stock_data(directory='/home/prasojo/belajar/BelajarData/dataset/saham/emiten', pic_directory = '/home/prasojo/belajar/BelajarData/dataset/saham/emiten_pic'):
    for filename in os.listdir(directory):
        file_name = f'{directory}/{filename}'
        df_saham = pd.read_excel(file_name)
        df_saham = df_saham.reindex(index=df_saham.index[::-1])
        df_saham = df_saham.set_index(pd.DatetimeIndex(df_saham['date'].values))
        emiten = filename.split('.')[0]
        pic_file_name = f'{pic_directory}/{emiten}.jpg'
        save = dict(fname=pic_file_name,dpi=100)
        mpf.plot(df_saham,type='candle',mav=(3,7),volume=True,figsize=(25,15),savefig=save)

