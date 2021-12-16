import csv

import requests
import json


def getHTMLText(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except:
        return "error in getHTMLText"


base = "https://v0.yiketianqi.com/api?version=history&appid=23295957&appsecret=qD0oO1SD&city="
date = "&year=2021&month=3"
cities = ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水']
with open('data.csv', 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ['city', 'ymd', 'bWendu', 'yWendu', 'tianqi', 'fengxiang', 'fengli', 'aqi', 'aqiInfo', 'aqiLevel'])
    for city in cities:
        url = str(base + city + date)
        data = json.loads(getHTMLText(url))
        for i in range(0, len(data['data'])):
            writer.writerow(
                [data['city'], data['data'][i]['ymd'], data['data'][i]['bWendu'], data['data'][i]['yWendu'],
                 data['data'][i]['tianqi'], data['data'][i]['fengxiang'], data['data'][i]['fengli'],
                 data['data'][i]['aqi'], data['data'][i]['aqiInfo'], data['data'][i]['aqiLevel']])
        print(data)