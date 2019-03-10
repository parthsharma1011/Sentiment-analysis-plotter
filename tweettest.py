#!/user/PycharmProjects/KOKO/sentiment_analyser.py


py__author__ = "Parth Sharma"




import tweepy,csv
import matplotlib.pyplot as plt
from sentiment_analyser import *
from time import time
import ray


class TwitterPlot:

    def __init__(self):
        self.tweets = []
        self.tweetText = []
        self.positive = 0
        self.negative = 0
        self.neutral = 0
        self.no_tweets = 0
        self.term_search = 0


    def DownloadData(self):
        '''
        :return: Add authentication keys to access twitter api, return tweets saved in  extweets.csv
        '''

        # authenticating with api credentials
        consumerKey = 'ADD KEYS HERE '
        consumerSecret = 'ADD KEYS HERE '
        accessToken = 'ADD KEYS HERE '
        accessTokenSecret = 'ADD KEYS HERE '
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        # input for term to be searched and how many tweets to search
        self.term_search = input("Enter Keyword/Tag (eg : tesla): ")
        self.no_tweets = int(input("Enter no of tweets to search on : "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=self.term_search, lang = "en").items(self.no_tweets)
        csvFile = open('extweets.csv', 'a')
        csvWriter = csv.writer(csvFile)

        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average of how people are reacting
        self.positive = self.percentage(self.positive, self.no_tweets)
        self.negative = self.percentage(self.negative, self.no_tweets)
        self.neutral = self.percentage(self.neutral, self.no_tweets)


    def cleanTweet(self, tweet):
        '''
        :param tweet: Extracted tweets
        :return: Clean tweets, removed of punctuations, special characters etc.
        '''
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    def percentage(self, part, whole):
        '''
        :param part: Part of the calculation
        :param whole: Overall value in the calculation
        :return:
        '''
        temp = 100 * float(part) / float(whole)
        return float(round(temp,2))

    def plotPieChart(self, positive, negative, neutral,term_search,no_tweets):
        '''
        :param positive: No of Positive sentiments
        :param negative: No of Negative sentiments
        :param neutral: No of Neutral sentiments
        :param term_search: Twitter tag requested by the user to be searched
        :param no_tweets: No of tweet search requested by the user
        :return:
        '''
        posper = self.percentage(positive,self.no_tweets)
        negper = self.percentage(negative,self.no_tweets)
        neuper = self.percentage(neutral,self.no_tweets)

        labels = ['Positive [' + str(posper) + '%]', 'Negative [' + str(negper) + '%]',
                  'Neutral [' + str(neuper) + '%]']

        sizes = [positive, negative, neutral]
        colors = ['#1FDA9A', '#DB3340', '#3A9AD9']  #['yellowgreen', 'red', 'gold']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        #plt.title('How people are reacting on ' + '#' + str(term_search).upper() + ' by analyzing ' + str(no_tweets) + ' Tweets.')
        plt.title('How people are reacting on #{} by analyzing [{}] tweets'.format(str(term_search).upper(),str(no_tweets)))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def updateValue(self,z):
        '''
        :param z: Calculated sentiment
        :return: Updated count
        '''
        if z == 'positive':
            self.positive += 1
        elif z == 'neutral':
            self.neutral += 1
        else:
            self.negative += 1


#TODO: Original code without multi-process takes longer to process
# if __name__== "__main__":
#
#     sa = TwitterPlot()
#     sa.DownloadData()
#     obj = SentimentAnalysis()
#
#     ST = time()
#
#     for tweet in sa.tweetText:
#         y = str(tweet,'utf-8')
#         z = obj.predict_sentiment(y)
#         print('for {} the sentiment is {}'.format(y, z))
#         sa.updateValue(z)
#
#     print( time() - ST )
#
#     sa.plotPieChart(sa.positive,sa.negative,sa.neutral,sa.term_search,sa.no_tweets)


if __name__ == '__main__':
    ray.init()
    sa = TwitterPlot()
    sa.DownloadData()
    obj = SentimentAnalysis()
    ST = time()
    rayResults = [ray.put(obj.predict_sentiment(str(tweet,'utf-8'))) for tweet in sa.tweetText]
    results = ray.get(rayResults)
    for arr in results:
        #print( arr[0] )
        sa.updateValue(arr[1])
    print(time()-ST)
    sa.plotPieChart(sa.positive,sa.negative,sa.neutral,sa.term_search,sa.no_tweets)

