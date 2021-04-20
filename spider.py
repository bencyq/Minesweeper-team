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
    url = "https://v0.yiketianqi.com/api?version=history&appid=79832935&appsecret=2qocDt9D&city=%E6%B5%99%E6%B1%9F&year=2018&month=5"
    text = getHTMLText(url)
    data = json.loads(text)
    print(data)

main()