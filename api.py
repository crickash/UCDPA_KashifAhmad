import requests
import matplotlib.pyplot as plt

dates = ["2020-05-18", "2020-06-18", "2020-07-18", "2020-08-18", "2020-09-18", "2020-10-18", "2020-11-18", "2020-12-18", "2021-01-18", "2021-02-18", "2021-03-18", "2021-04-18"]

gbp = []
cad = []
usd = []
aud = []


for date in dates:
    request = requests.get("https://api.ratesapi.io/api/"+date)
    data = request.json()
    gb = data['rates']["GBP"]
    ca = data['rates']["CAD"]
    us = data['rates']["USD"]
    au = data['rates']["AUD"]
    gbp.append(gb)
    cad.append(ca)
    usd.append(us)
    aud.append(au)


plt.figure(figsize=(15,5))

plt.plot(dates, gbp, marker='o')
plt.plot(dates, cad, marker='o')
plt.plot(dates, usd, marker='o')
plt.plot(dates, aud, marker='o')

plt.legend(["GBP", "CAD", "USD", "AUD"])
plt.show()