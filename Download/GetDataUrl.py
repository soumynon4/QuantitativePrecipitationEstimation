# @Time     : 7/10/2021 9:14 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : GetDataUrl.py
# @Project  : QuantitativePrecipitationEstimation
import os.path

import requests
from lxml import etree


def isLeapYear(year: int):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0) or year % 172800 == 0:
        return True
    else:
        return False


def getUrlList(url):
    header = {"User-Agent": 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
    html_text = requests.get(url=url, headers=header).text
    html = etree.HTML(html_text)
    urlList = html.xpath(
        '//body/div[@id="background"]/div[@id="wrap"]/div[@id="content"]/div[@class="pad"]/div[@class="col-2-right-700"]/div[@class="bdpSection"]/div[@class="bdpLink"]/a/@href')
    return urlList


def urlForYearMonthDay(radarStation: str, year: int, month: int, day: int, SavingDirectory: str = ".", isReturn: bool = False):
    if year <= 0 or month <= 0 or month > 12 or day <= 0 or day > 31:
        raise ValueError

    requestsWeb = "https://www.ncdc.noaa.gov/nexradinv/bdp-download.jsp?id={}&yyyy={:4}&mm={:02}&dd={:02}&product=AAL2".format(
        radarStation.upper(), year, month, day)

    totalHttp = getUrlList(requestsWeb)
    totalHttp = list(map(lambda url: str(url) + '\n', totalHttp))
    print(len(totalHttp))

    if isReturn:
        return totalHttp

    with open(os.path.join(SavingDirectory, "{}_{}_{:02}_{:02}.log".format(radarStation.upper(), year, month, day)),
              "w") as f:
        f.writelines(totalHttp)


def urlForYear(radarStation: str, year: int, SavingDirectory: str = "."):
    ret = []
    days = 0
    for month in range(12):
        if month + 1 in {1, 3, 5, 7, 8, 10, 12}:
            days = 31
        elif month + 1 in {4, 6, 9, 11}:
            days = 30
        elif month + 1 == 2 and isLeapYear(year):
            days = 29
        else:
            days = 28

        for day in range(days):
            requestsWeb = "https://www.ncdc.noaa.gov/nexradinv/bdp-download.jsp?id={}&yyyy={:4}&mm={:02}&dd={:02}&product=AAL2".format(
                radarStation.upper(), year, month + 1, day + 1)
            Http = getUrlList(requestsWeb)
            Http = list(map(lambda url: str(url) + '\n', Http))
            ret.extend(Http)

    with open(os.path.join(SavingDirectory, "{}_{}.log".format(radarStation.upper(), year)),
              "w") as f:
        f.writelines(ret)
    print(len(ret))


def getDateForGauge(radarStation: str, GaugeFilePath: str, threshold: int, SavingDirectory: str = "."):
    gauges = sorted(os.listdir(GaugeFilePath))
    ans = set()
    for gauge in gauges:
        with open(os.path.join(GaugeFilePath, gauge)) as f:
            metadata = f.readline()
            rain_gauge_range = float(metadata.split()[-5])
            if rain_gauge_range <= 100.0:
                for record in f:
                    content = record.split()
                    if int(content[-1]) >= threshold and float(content[7]) > 0.0:
                        date = content[0] + content[1] + content[2] + content[4]
                        ans.add(date)
            else:
                print("{} out of 100 km".format(gauge))
                pass

    ans = sorted(ans)
    print(ans)

    urls = list()
    temporaryUrls = None
    for date in ans:
        print(date[0:4], date[4:6], date[6:8], date[8:10])
        if temporaryUrls is None:
            temporaryUrls = urlForYearMonthDay(radarStation, int(date[0:4]), int(date[4:6]), int(date[6:8]),
                                               SavingDirectory, isReturn=True)
        elif len(temporaryUrls) == 0:
            print("Because Empty Urls so Request")
            temporaryUrls = urlForYearMonthDay(radarStation, int(date[0:4]), int(date[4:6]), int(date[6:8]),
                                               SavingDirectory, isReturn=True)

        elif temporaryUrls[0].split('.')[-1].split('/')[1] == date[0:4] and temporaryUrls[0].split('.')[-1].split('/')[2] == date[4:6] and temporaryUrls[0].split('.')[-1].split('/')[3] == date[6:8]:
            print("Because Same Year Month Day so Pass")
            pass

        else:
            print("Because Different Year Month Day so Request")
            temporaryUrls = urlForYearMonthDay(radarStation, int(date[0:4]), int(date[4:6]), int(date[6:8]),
                                               SavingDirectory, isReturn=True)

        for tmp in temporaryUrls:
            if date[8:10] == tmp.split('.')[-1].split('/')[-1].split('_')[1][0:2]:
                print(date)
                print(tmp)
                urls.append(tmp)
            else:
                continue

    print(len(urls))

    with open(os.path.join(SavingDirectory, "{}_{}.log".format(radarStation.upper(), ans[0][0:4])), "w") as f:
        f.writelines(urls)


if __name__ == '__main__':
    getDateForGauge("KMLB", "c:\\users\\admin\\downloads\\GMIN_SFL_2021", 30, "c:\\users\\admin\\desktop")

