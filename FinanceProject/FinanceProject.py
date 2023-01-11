from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

#Data
#We need to get data using pandas datareader. We will get stock information for the following banks:
#Bank of America
#CitiGroup
#Goldman Sachs
#JPMorgan Chase
#Morgan Stanley
#Wells Fargo
#Figure out how to get the stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks. Set each bank to be a separate dataframe,
#with the variable name for that bank being its ticker symbol. This will involve a few steps:**
#1. Use datetime to set start and end datetime objects.
#2. Figure out the ticker symbol for each bank.
#3. Figure out how to use datareader to grab info on the stock.
#Use [this documentation page](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html) for hints and instructions
#(it should just be a matter of replacing certain values. Use google finance as a source, for example:

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

#Bank of America
BAC = data.DataReader("BAC", 'google', start, end)

#CitiGroup
C = data.DataReader("C", 'google', start, end)

#Goldman Sachs
GS = data.DataReader("GS", 'google', start, end)

#JPMorgan Chase
JPM = data.DataReader("JPM", 'google', start, end)

#Morgan Stanley
MS = data.DataReader("MS", 'google', start, end)

#Wells Fargo
WFC = data.DataReader("WFC", 'google', start, end)

#Could also do this for a Panel Object
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'google', start, end)

#Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

#Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)

#Set the column name levels (this is filled out for you):
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

#Check the head of the bank_stocks dataframe.**
bank_stocks.head()

#EDA
#What is the max Close price for each bank's stock throughout the time period?
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()

# Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock.
returns = pd.DataFrame()

#We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for
#each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()

#Create a pairplot using seaborn of the returns dataframe.
import seaborn as sns
sns.pairplot(returns[1:])

#Background on [Citigroup's Stock Crash available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 
#You'll also see the enormous crash in value if you take a look a the stock price plot (which we do later in the visualizations.)
#Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. 
#You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?

# Worst Drop (4 of them on Inauguration day)
returns.idxmin()

#You should have noticed that Citigroup's largest drop and biggest gain were very close to one another, did anythign significant happen in that time frame? 
#[Citigroup had a stock split.](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may)
#Best Single Day Gain
#citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()

#Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?
returns.std() # Citigroup riskiest
returns.ix['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA

#Create a distplot using seaborn of the 2015 returns for Morgan Stanley 
sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)

#Create a distplot using seaborn of the 2008 returns for CitiGroup 
sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


#More Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()

#Create a line plot showing Close price for each bank for the entire index of time. (Hint: Try using a for loop, or use
#[.xs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html) to get a cross section of the data.)
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()
# plotly
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()

#Moving Averages 
#Let's analyze the moving averages for these stocks in the year 2008. 
#Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008
plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()

#Create a heatmap of the correlation between the stocks Close Price.**
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

#Use seaborn's clustermap to cluster the correlations together:**
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


#In this second part of the project we will rely on the cufflinks library to create some Technical Analysis plots. 
#Use .iplot(kind='candle) to create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016.**
BAC[['Open', 'High', 'Low', 'Close']].ix['2015-01-01':'2016-01-01'].iplot(kind='candle')

#Use .ta_plot(study='sma') to create a Simple Moving Averages plot of Morgan Stanley for the year 2015.**
MS['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')

#Use .ta_plot(study='boll') to create a Bollinger Band Plot for Bank of America for the year 2015.**
BAC['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')


