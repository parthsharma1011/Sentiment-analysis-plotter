#!/user/PycharmProjects/KOKO/sentiment_analyser.py


py__author__ = "Parth Sharma"



import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
import pickle
import os
from imblearn.over_sampling import RandomOverSampler
import re
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.pipeline import Pipeline as imbPipeline
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


DATA_PATH=os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(DATA_PATH,"data")


class SentimentAnalysis :
    """
    Sentiment analysis class

    ARGS:
        Params: contains pickle model import,tweets data loading
        Count normalizing, training and prediction
    """

    def __init__(self):
        """
        Load the pretrained pickle file
        Args:
            Params: pickle file on tweets.csv
        """
        
        try:
            self.filename = 'finalized_model.pkl'
            self.clf = pickle.load(open(self.filename, 'rb'))
        except:
            self.clf = self.train_model()
    
    def load_dataset_samples(self):
        """

        Args:
            Params: Tweets.csv
        :return: data frame (Normalized with normalizer function)
        """

        _df = pd.read_csv(os.path.join(DATA_PATH,"Tweets.csv"),encoding='latin-1')
        col_name = ['sentiment','text']
        df = pd.DataFrame(columns= col_name)
        df['sentiment'] = _df.airline_sentiment
        df['text'] = _df.text
        df = shuffle(df)
        df.reset_index(drop= True, inplace= True)
        df['normalized_tweet'] = df.text.apply(self.normalizer)
        return df

    def normalizer(self, tweet):
        """

        :param tweet: Individual tweet block lemmatized
        :return: tweets lemma (lemmatized tweet block )
        """
        only_letters = re.sub("[^a-zA-Z]", " ", tweet)
        only_letters = only_letters.lower()
        only_letters = only_letters.split()
        filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
        lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
        lemmas = ' '.join(lemmas)
        return lemmas

    def train_model(self):
        print("Training model....")
        df = self.load_dataset_samples()
        clf = imbPipeline([
                        ('vect', TfidfVectorizer(ngram_range=(1,2))),
                        ('tfidf', TfidfTransformer(use_idf=False)),
                        ('over-sampling', RandomOverSampler()),
                        ('clf', LogisticRegression(multi_class='multinomial', solver='newton-cg'))
                        ])
        
        clf.fit(df.normalized_tweet, df.sentiment)
        pickle.dump(clf, open(self.filename, 'wb'))
        return clf

    def predict_sentiment(self, query):
        #normalized_query = self.normalizer(query)
        prob_cs = self.clf.predict_proba([query])[0]
        #print(prob_cs)  # hash out
        if prob_cs[0] > 0.60:
            z = 'negative'
        elif prob_cs[1] > 0.60:
            z = 'neutral'
        elif prob_cs[2] > 0.60:
            z = 'positive'
        else:
            z = 'neutral'

        return [ 'for {} the sentiment is {}'.format(query, z), z]

    def train_model_with_cv(model, params, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Use Train data to parameter selection in a Grid Search
        gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
        gs_clf = gs_clf.fit(X_train, y_train)
        model = gs_clf.best_estimator_

        # Use best model and test data for final evaluation
        y_pred = model.predict(X_test)

        _f1 = f1_score(y_test, y_pred, average='micro')
        _confusion = confusion_matrix(y_test, y_pred)
        __precision = precision_score(y_test, y_pred)
        _recall = recall_score(y_test, y_pred)
        _statistics = {'f1_score': _f1,
                       'confusion_matrix': _confusion,
                       'precision': __precision,
                       'recall': _recall
                       }

        return model, _statistics


if __name__ == '__main__':
    obj = SentimentAnalysis()
    while True:
        user_input= input("Enter: ")
        if user_input == 'quit':
            break
        print(obj.predict_sentiment(user_input))
            
            





