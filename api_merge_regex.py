# Merge DataFrames

# imports
import requests
import matplotlib.pyplot as plt
import pandas as pd

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


#=======================================================================


# Set up dictionary for each date according to its value for GBP currency
data_gbp = {'date':  dates,
        'GBP': gbp}

# Set up dictionary for each date according to its value for USD currency
data_usd = {'date':  dates,
        'USD': usd}

# Create DataFrames for both dictionaries
df1 = pd.DataFrame (data_gbp, columns = ['date','GBP'])
df2 = pd.DataFrame (data_usd, columns = ['date','USD'])


# Merge GBP and USD DataFrames on the "date" Column
merged_dataframe = df1.merge(df2, on='date')

# Show merged DataFrame
print(merged_dataframe)

#=======================================================================

#RegEx

# Import the re module
import re

string = "The fall of #Watson #Health: How #IBM’s plan to change the face of healthcare with #AI fell apart. Great read by @caseymross @mariojoze @statnews @GersonRolim @intellimetri"


# Write the regex
users = r"@[A-Za-z0-9\W]+"
hastags = r"\#\w+"

# Find all matches of regex
print(re.findall(users, string))
print(re.findall(hastags, string))