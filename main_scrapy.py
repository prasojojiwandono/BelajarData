import scrapy

class PostsSpider(scrapy.Spider):
    name = 'posts'

    start_url = [
        'https://finance.yahoo.com/quote/AALI.JK/history?p=AALI.JK'
    ]

    def parse(self, response):
        xx = response.body
        if xx :
            print('ada')
            print(type(xx))
            