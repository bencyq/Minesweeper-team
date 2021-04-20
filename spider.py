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


def main():
    base = "https://v0.yiketianqi.com/api?version=history&appid=79832935&appsecret=2qocDt9D&city="
    date = "&year=2021&month=3"
    cities = ['杭州','宁波','温州','嘉兴','湖州','绍兴','金华','衢州','舟山','台州','丽水']
    for city in cities:
        url = str(base+city+date)
        data=getHTMLText(url)
        print(data)



main()