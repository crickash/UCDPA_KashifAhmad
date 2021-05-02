# imports
import requests
import matplotlib.pyplot as plt

# Set up dates list to retrieve currency conversion rate for these dates
dates = ["2020-05-18", "2020-06-18", "2020-07-18", "2020-08-18", "2020-09-18", "2020-10-18", "2020-11-18", "2020-12-18", "2021-01-18", "2021-02-18", "2021-03-18", "2021-04-18"]

# Initialize arrays for currencies
gbp = []
cad = []
usd = []
aud = []

# Request API for each dat
for date in dates:
    request = requests.get("https://api.ratesapi.io/api/"+date)

    # convert raw data into JSON format
    data = request.json()

    # Retrieve values for each currency
    gb = data['rates']["GBP"]
    ca = data['rates']["CAD"]
    us = data['rates']["USD"]
    au = data['rates']["AUD"]

    # Append values to the respective arrays
    gbp.append(gb)
    cad.append(ca)
    usd.append(us)
    aud.append(au)


# Display chart of the rates for the currencies
plt.figure(figsize=(15,5))

plt.plot(dates, gbp, marker='o')
plt.plot(dates, cad, marker='o')
plt.plot(dates, usd, marker='o')
plt.plot(dates, aud, marker='o')

plt.legend(["GBP", "CAD", "USD", "AUD"])
plt.show()