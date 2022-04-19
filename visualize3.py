import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime
import dateutil
import numpy as np

## NEXT STEP:
    ## 1) Organize Code with pseudocode
    ## 2) Publish on Github or not

#Goal:
    # 1) display number of moderna_tweets and sinovac_tweets over time.

    # 2) display avg_likes_per_post over time of moderna_tweets and sinovac_tweets

# Data Cleaning
# 1) for Goal (1): only need variables (date, dummy_for_moderna, dummy_for_sinovac)
# 2) for Goal (2): only need variables (date, likes, dummy_for_moderna, dummy_for_sinovac)

# To-do 1) read and combine moderna.csv and sinovac.csv into vax.csv
df1 = pd.read_csv("moderna.csv",header = None )
df1["vax"] = "moderna" #assign identifier

df2 = pd.read_csv("sinovac.csv",header = None)
df2["vax"] = "sinovac" #assign identifier

df0 = pd.read_csv("pfizer.csv",header = None)
df0["vax"] = "pfizer" #assign identifier

df4 = pd.read_csv("sputnik.csv",header = None)
df4["vax"] = "sputnik" #assign identifier

df3 = pd.concat([df1,df2,df0,df4]) #merge into 1 df

# To-do 2) eliminate unnecessary variables by subsetting.
df = df3[[3,17,"vax"]] #subset that includes only (date,retweets,vaccine_identifier)
df.columns = ['date','retweets','vax']
# To-do 3) remove all data that have 0 retweets by conditional subsetting
df = df[df['retweets'].astype(str).str.isdigit() == True]
df['retweets'] = df['retweets'].astype(int)
df = df[df['retweets'] <= 200]


# To-do 4) limit dates between 4-01 to 4-18 by subsetting
df = df[df['date'] >= '2022-03-31']
df = df[df['date'] <= '2022-04-18']

# To-do 5) change date variable from string to dateFrame
df['date'] = df['date'].apply(dateutil.parser.parse)

#To-do 6) create new dataframe for aggregating moderna
df_moderna_count = df[df['vax'] == 'moderna']
df_moderna_count = df_moderna_count[['date','vax']]

df_sinovac_count = df[df['vax'] == 'sinovac']
df_sinovac_count = df_sinovac_count[['date','vax']]

df_pfizer_count = df[df['vax'] == 'pfizer']
df_pfizer_count = df_pfizer_count[['date','vax']]

df_sputnik_count = df[df['vax'] == 'sputnik']
df_sputnik_count = df_sputnik_count[['date','vax']]
print(df_sputnik_count)

x = [df_moderna_count['date']]


a = pd.crosstab(index=df_moderna_count['date'], columns='count')
b = pd.crosstab(index=df_sinovac_count['date'], columns='count')
c = pd.crosstab(index=df_pfizer_count['date'], columns='count')
d = pd.crosstab(index=df_sputnik_count['date'], columns='count')

q = sum(a['count'])
w = sum(b['count'])
e = sum(c['count'])
r = sum(d['count'])
t = np.array([q,w,e,r])
mlabels = ["Moderna", "Sinovac", "Pfizer", "Sputnik"]
mcolors = ["blue","red","green","gray"]

plt.pie(t,labels  = mlabels, shadow = True,autopct='%1.1f%%')
plt.title("Pie Chart Showing Share of Tweets That Mentions Moderna, Sinovac, Pfizer, and Sputnik in the Philippines")
plt.show()


y = [df_sinovac_count['date']]
z = [df_pfizer_count['date']]
w = [df_sputnik_count['date']]

plt.style.use('seaborn-deep')
plt.tight_layout()

plt.plot(a,alpha = 0.5, label = 'moderna', color = 'blue')
plt.plot(b,alpha = 0.5, label = 'sinovac', color = 'red')
plt.plot(c,alpha = 0.5, label = 'pfizer', color = 'green')
plt.plot(d,alpha = 0.5, label = 'sputnik', color = 'gray')

# plt.hist(x,label = 'moderna', edgecolor = 'black', rwidth = 2.5, linewidth = 1.2, alpha = 0.4, color = 'blue')
# plt.hist(y,label = 'sinovac',edgecolor = 'black', rwidth = 2.5, linewidth = 1.2, alpha = 0.4, color = 'red')
# plt.hist(z,label = 'pfizer', edgecolor = 'black', rwidth = 2.5, linewidth = 1.2, alpha = 0.4, color = 'green')
# plt.hist(w,label = 'sputnik',edgecolor = 'black', rwidth = 2.5, linewidth = 1.2, alpha = 0.4, color = 'gray')

plt.legend(loc='upper right')
plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)
plt.xticks(rotation=75, ha="right",fontsize = 5)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.title("Number of Tweets That Exclusively Mention Either Moderna, Sinovac, Pfizer, and Sputnik Tagged in the Philippines")
plt.grid()
plt.show()

# Next: graph avg_retweets_per_post_per_day
# 1) find average retweets per post per day
df_moderna_count = df[df['vax'] == 'moderna']
df_moderna_retweets = df_moderna_count[['date','retweets']]

df_sinovac_count = df[df['vax'] == 'sinovac']
df_sinovac_retweets = df_sinovac_count[['date','retweets']]

df_pfizer_count = df[df['vax'] == 'pfizer']
df_pfizer_retweets = df_pfizer_count[['date','retweets']]

df_sputnik_count = df[df['vax'] == 'sputnik']
df_sputnik_retweets = df_sputnik_count[['date','retweets']]


#create dictionary to store retweet averages
retweet_avg_moderna = {'date': [], 'retweets': []}

# Split the data into different regions
for date in df_moderna_retweets['date'].unique():
    tempdf = df_moderna_retweets[df_moderna_retweets['date'] == date]

    # Apply an aggregation function
    average = tempdf['retweets'].mean()

    # Combine the data into a DataFrame
    retweet_avg_moderna['date'].append(date)
    retweet_avg_moderna['retweets'].append(average)

aggregate_df_moderna = pd.DataFrame(retweet_avg_moderna)

## Do the same for sinovac df
retweet_avg_sinovac = {'date': [], 'retweets': []}
# Split the data into different regions
for date in df_sinovac_retweets['date'].unique():
    tempdf = df_sinovac_retweets[df_sinovac_retweets['date'] == date]

    # Apply an aggregation function
    average = tempdf['retweets'].mean()

    # Combine the data into a DataFrame
    retweet_avg_sinovac['date'].append(date)
    retweet_avg_sinovac['retweets'].append(average)

aggregate_df_sinovac = pd.DataFrame(retweet_avg_sinovac)


##
## Do the same for pfizer df
retweet_avg_pfizer = {'date': [], 'retweets': []}
# Split the data into different regions
for date in df_pfizer_retweets['date'].unique():
    tempdf = df_pfizer_retweets[df_pfizer_retweets['date'] == date]

    # Apply an aggregation function
    average = tempdf['retweets'].mean()

    # Combine the data into a DataFrame
    retweet_avg_pfizer['date'].append(date)
    retweet_avg_pfizer['retweets'].append(average)

aggregate_df_pfizer = pd.DataFrame(retweet_avg_pfizer)

##
## Do the same for pfizer df
retweet_avg_sputnik = {'date': [], 'retweets': []}
# Split the data into different regions
for date in df_sputnik_retweets['date'].unique():
    tempdf = df_sputnik_retweets[df_sputnik_retweets['date'] == date]

    # Apply an aggregation function
    average = tempdf['retweets'].mean()

    # Combine the data into a DataFrame
    retweet_avg_sputnik['date'].append(date)
    retweet_avg_sputnik['retweets'].append(average)

aggregate_df_sputnik = pd.DataFrame(retweet_avg_sputnik)

# print(aggregate_df_moderna)
# print('sinovac below \n')
# print(aggregate_df_sinovac)

plt.plot(aggregate_df_moderna['date'], aggregate_df_moderna['retweets'], alpha=0.4,label = 'moderna')
plt.plot(aggregate_df_sinovac['date'], aggregate_df_sinovac['retweets'], alpha=0.4,label = 'sinovac')
plt.plot(aggregate_df_pfizer['date'], aggregate_df_pfizer['retweets'], alpha=0.4,label = 'pfizer')
plt.plot(aggregate_df_sputnik['date'], aggregate_df_sputnik['retweets'], alpha=0.4,label = 'sputnik')

plt.legend(loc='upper right',fontsize = 7)
plt.xlabel("Date", size=14)
plt.ylabel("Average Retweets Per Day", size=14)
plt.xticks(rotation=75, ha="right",fontsize = 5)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.title("Average Retweets with Mentions of Moderna and Sinovac Tagged in the Philippines")
plt.grid()
plt.show()

# df = pd.concat(map(pd.read_csv, ['moderna.csv','sinovac.csv']))

# file1 = pd.read_csv("moderna.csv")
# file2 = pd.read_csv("sinovac.csv")

# merged = file1.merge(file2

    # only need (date,likes,)

# To-do 1) Add headers for relevant variables
# x = []  
# for row in df:
#     x.append(row[3])
#     y.append(row[])
# print(x)
# for row in plots:
    #     x.append(row[2])
    #     y.append(row[1])
#plt.scatter(x, y)
#plt.xticks(rotation = 25)
# plt.xlabel('Names')
# plt.ylabel('Values')
# plt.title('Patients Blood Pressure Report', fontsize = 10)
  
# plt.show()