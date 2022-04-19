import twint

c = twint.Config()

c.Search = "sputnik"     # topic
c.Limit = 1000      # number of Tweets to scrape
c.Store_csv = True       # store tweets in a csv file
c.Output = "/Users/aldrin/Desktop/tweet/sputnik.csv"     # path to csv file
c.Near = "Philippines"


#c.Since = "2020-01-01"
#c.Until = "2021-12-31"
#c.Until = "2022-01-01"
#c.Verified = True
#c.Min_replies = 1

#run code
twint.run.Search(c)