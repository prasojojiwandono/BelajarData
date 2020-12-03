import requests
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import xlrd
from xlwt import Workbook 
import plotly.graph_objects as go
import plotly
import pandas as pd

# percobaan


 
loc = ("/home/prasojo/belajar/BelajarData/dataset/saham/JII.xlsx")
 
wb_jii_list = xlrd.open_workbook(loc)
sheet_jii_list = wb_jii_list.sheet_by_index(0)

 
for i in range(sheet_jii_list.nrows):
    emiten = sheet_jii_list.cell_value(i, 0)

wb_analyst = Workbook() 
# wb_chart = Workbook() 
  
sheet_analyst = wb_analyst.add_sheet('Sheet 1') 
# sheet_chart = wb_chart.add_sheet('Sheet 1')
   
sheet_analyst.write(0, 0, 'JII') 
sheet_analyst.write(0, 1, 'date') 
sheet_analyst.write(0, 2, 'Open') 
sheet_analyst.write(0, 3, 'High') 
sheet_analyst.write(0, 4, 'Low') 
sheet_analyst.write(0, 5, 'Close') 
sheet_analyst.write(0, 6, 'Volume')

# sheet_chart.write(0, 0, 'JII') 
# sheet_chart.write(0, 1, 'date') 
# sheet_chart.write(0, 2, 'Open') 
# sheet_chart.write(0, 3, 'High') 
# sheet_chart.write(0, 4, 'Low') 
# sheet_chart.write(0, 5, 'Close') 
# sheet_chart.write(0, 6, 'Volume')
  

def create_stock_chart(df_saham, emiten):
    df_saham = df_saham.reindex(index=df_saham.index[::-1])
    df_saham = df_saham.set_index(pd.DatetimeIndex(df_saham['date'].values))
    figure = go.Figure(
                data =[
                    go.Candlestick(
                    x = df_saham.index,
                    low = df_saham['Low'],
                    high = df_saham['High'],
                    close = df_saham['Close'],
                    open = df_saham['Open'],
                    increasing_line_color = 'green',
                    decreasing_line_color = 'red',
                    )
                ],
                layout=go.Layout(
                    title=go.layout.Title(text=emiten)
                )
            )
    figure.write_image(file = '/home/prasojo/belajar/BelajarData/dataset/saham/'+emiten+'.png',
                  width = 1600,
                  height = 1000)

# def time_code():
#     perhari = 86400
#     nov22 = 1606003200
#     tgl_nov22 = datetime(2020,11,22)
#     delta = datetime.now() - tgl_nov22
#     delta_day = delta.days
#     kode_now = nov22 + (delta_day * perhari)
#     kode_past = kode_now - (perhari * 140)
#     return str(kode_now), str(kode_past)

# kode_now, kode_past = time_code()

baris = 1
baris_analyst = 1
for i in range(sheet_jii_list.nrows):
    pd_data = []
    emiten = sheet_jii_list.cell_value(i, 0)
    print(f'proses {emiten} {i}/{sheet_jii_list.nrows}')
    source_html = 'https://finance.yahoo.com/quote/'+emiten+'.JK/history?p='+emiten+'.JK'
    src = requests.get(source_html).text
    soup = BeautifulSoup(src,'lxml')
    table = soup.findAll('table')[0]
    body = table.findAll('tbody')[0]
    rows = body.findAll('tr')
    uu = 0
    characters_to_remove = [',','.00']
    time_format = '%b %d, %Y' 
    baris_emiten = 1
    for row in rows:
        pd_data_kolom = []
        kolom = row.findAll('td')
        count = 0
        for data in kolom:
            if len(kolom) < 6 :
                break

            volume = kolom[6].text
            if volume == '-':
                break

            volume = kolom[6].span.text
            if len(str(volume)) <= 2:
                break


            if count > 0 :
                if count != 5 :
                    value = data.span.text
                    for character in characters_to_remove:
                        value = value.replace(character, '')
                    if count > 5:
                        if baris_emiten <= 30:
                            sheet_analyst.write(baris_analyst, 0, emiten)
                            sheet_analyst.write(baris_analyst, count, int(value)) 
                            baris_analyst = baris_analyst +1

                        # sheet_chart.write(baris, 0, emiten)
                        # sheet_chart.write(baris, count, value) 
                        pd_data_kolom.append(value)
                        baris = baris + 1
                        baris_emiten = baris_emiten + 1
                    else:
                        # sheet_chart.write(baris, count+1, value)
                        pd_data_kolom.append(value) 

                        if baris_emiten <= 30:
                            sheet_analyst.write(baris_analyst, count+1, int(value)) 


            else:
                value = data.span.text
                value = datetime.strptime(value, time_format) 
                value = datetime.strftime(value, '%Y-%m-%d')                
                # sheet_chart.write(baris, count+1, value) 
                pd_data_kolom.append(emiten)
                pd_data_kolom.append(value)

                if baris_emiten <= 30:
                    sheet_analyst.write(baris_analyst, count+1, value) 

            count = count + 1

        if count > 0:
            pd_data.append(pd_data_kolom)

        if baris_emiten > 90:
            df_saham = pd.DataFrame(pd_data,
                columns=['JII', 'date', 'Open','High','Low','Close','Volume'])
            
            create_stock_chart(df_saham, emiten)
            break


wb_analyst.save('/home/prasojo/belajar/BelajarData/dataset/saham/JII_analysis.xlsx') 
# wb_chart.save('/home/prasojo/belajar/BelajarData/dataset/JII_chart.xlsx')








# source_html = '/home/prasojo/belajar/javascript/dom_exercise/index.html'

# with open(source_html, 'r') as html_file:
#     content = html_file.read()
#     soup = BeautifulSoup(content, 'lxml')
#     # print(soup.prettify())
#     tag = soup.find('li')
#     tags = soup.find_all('li')
#     for x in tags:
#         print(x.text)