# imports
import requests
import matplotlib.pyplot as plt
import pandas as pd
import re

#====================API====================
# Set up dates list to retrieve currency conversion rate for these dates
dates = ["2020-05-18", "2020-06-18", "2020-07-18", "2020-08-18", "2020-09-18", "2020-10-18", "2020-11-18", "2020-12-18", "2021-01-18", "2021-02-18", "2021-03-18", "2021-04-18"]

# Initialize arrays for currencies
gbp = []
cad = []
usd = []
aud = []

# Request API for each date
for date in dates:
    """ the API usage limit is reset at the end of every month
    however it is not known if the API key is also reset
    therefore it is a possibility that API will not run if access key is also reset on 1st June """
    request = requests.get("http://api.exchangeratesapi.io/v1/"+date+"?access_key=48426be9e5c8e77f26b04c324cace3f2")

    # verify status code
    #print(request.status_code)

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
#plt.show()


#====================MERGE DATAFRAMES====================


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
#print(merged_dataframe)



#====================REGEX====================

string = "The fall of #Watson #Health: How #IBM’s plan to change the face of healthcare with #AI fell apart. Great read by @caseymross @mariojoze @statnews @GersonRolim @intellimetri"


# Write the regex
users = r"@[A-Za-z0-9\W]+"
hastags = r"\#\w+"

# Find all matches of regex
#print(re.findall(users, string))
#print(re.findall(hastags, string))