import json
import random
import pandas as pd

from textblob import TextBlob

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SentimentAnalysis:
    def __init__(self, tweets_file='data/preprocessed_tweets.json', labels_file='data/sentiment_labels.json', seed=42):
        # initialize with a seed for reproducibility
        random.seed(seed)
        # load data and create a TF-IDF vectorizer
        self.load_data(tweets_file, labels_file)
        self.tfidf_vectorizer = TfidfVectorizer()

    def load_data(self, tweets_file, labels_file):
        # load preprocessed tweets and labels
        with open(tweets_file, 'r', encoding='utf-8') as file:
            self.preprocessed_tweets = json.load(file)

        num_tweets_to_label = 600
        # randomly sample tweets for labeling
        self.sampled_tweets = random.sample(self.preprocessed_tweets, num_tweets_to_label)
        labeled_content = [tweet for tweet in self.sampled_tweets]
        
        # create a dataset of unlabeled tweets for use later
        self.unlabeled_tweets = [tweet if tweet not in labeled_content else labeled_content.remove(tweet) for tweet in self.preprocessed_tweets]

        with open(labels_file, 'r', encoding='utf-8') as file:
            sentiment_labels = json.load(file)

        # convert sentiment labels to numerical values
        self.labels = [2 if label == 'positive' else 1 if label == 'neutral' else 0 for label in sentiment_labels]

        labeled_data = pd.DataFrame({'tweets': [tweet for tweet in self.sampled_tweets], 'sentiment': self.labels})

        # split the labeled data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            labeled_data['tweets'], labeled_data['sentiment'], test_size=0.4, random_state=42
        )

    def train_classifier(self, method='naive_bayes'):
        if method == 'naive_bayes':
            self.train_naive_bayes()
        elif method == 'textblob':
            pass  # No training needed for TextBlob

    def train_naive_bayes(self):
        # use TF-IDF vectorizer to convert text data into numerical features
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)

        # train and fit the Naive Bayes classifier
        self.clf = MultinomialNB()
        self.clf.fit(X_train_tfidf, self.y_train)

    def analyze_sentiment_naive_bayes(self, comment):
        # use the trained Naive Bayes classifier to predict sentiment for specific comment
        return self.clf.predict(self.tfidf_vectorizer.transform([comment]))[0]
    
    def analyze_sentiment_textblob(self, comment):
        # analyze sentiment using TextBlob
        analysis = TextBlob(comment)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return 2  # positive
        elif polarity < 0:
            return 0  # negative
        else:
            return 1  # neutral

    def evaluate(self, method='naive_bayes'):
        # evaluate the performance of the specified method
        if method == 'naive_bayes':
            self.evaluate_naive_bayes()
        elif method == 'textblob':
            self.evaluate_textblob()

    def evaluate_naive_bayes(self):
        # use the trained Naive Bayes classifier to make predictions
        X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        y_pred = self.clf.predict(X_test_tfidf)

        # evaluate performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # print evaluation metrics
        print(f"Naive Bayes - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    def evaluate_textblob(self):
        sentiment_results = [self.analyze_sentiment_textblob(comment) for comment in self.sampled_tweets]

        # evaluate performance metrics
        accuracy = accuracy_score(self.labels, sentiment_results)
        precision = precision_score(self.labels, sentiment_results, average='weighted')
        recall = recall_score(self.labels, sentiment_results, average='weighted')
        f1 = f1_score(self.labels, sentiment_results, average='weighted')
        
        # print evaluation metrics
        print(f"TextBlob - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    def predict_unlabeled_data(self, method='naive_bayes'):
        # use Naive Bayes on unlabeled data
        if method == 'naive_bayes':
            X_unlabeled_tfidf = self.tfidf_vectorizer.transform(self.unlabeled_tweets)
            predictions = list(self.clf.predict(X_unlabeled_tfidf))

        # use TextBlob on unlabeled data
        elif method == 'textblob':
            predictions = [self.analyze_sentiment_textblob(tweet) for tweet in self.unlabeled_tweets]
        else:
            raise ValueError(f"Invalid method: {method}. Choose either 'NaiveBayes' or 'TextBlob'.")

        # calculate statistics for the specified method
        method_stats = {
            'Total': len(predictions),
            'Positive': predictions.count(2),
            'Neutral': predictions.count(1),
            'Negative': predictions.count(0),
            'Positive Percentage': (predictions.count(2) / len(predictions)) * 100,
            'Neutral Percentage': (predictions.count(1) / len(predictions)) * 100,
            'Negative Percentage': (predictions.count(0) / len(predictions)) * 100
        }

        # determine the sentiment with the highest percentage
        sentiment_percentages = {
            'Positive': method_stats['Positive Percentage'],
            'Neutral': method_stats['Neutral Percentage'],
            'Negative': method_stats['Negative Percentage']
        }

        highest_sentiment = max(sentiment_percentages, key=sentiment_percentages.get)

        # print a corresponding statement based on the highest sentiment
        if highest_sentiment == 'Positive':
            print("Yay, most people love this show!")
        elif highest_sentiment == 'Neutral':
            print("Seems most people are on the fence about the show.")
        elif highest_sentiment == 'Negative':
            print("Oh no, most people hate this show.")


        return method_stats


# Example usage
sentiment_analysis = SentimentAnalysis()
sentiment_analysis.train_classifier(method='naive_bayes')
sentiment_analysis.evaluate(method='naive_bayes')
sentiment_analysis.predict_unlabeled_data(method='naive_bayes')

sentiment_analysis.train_classifier(method='textblob')
sentiment_analysis.evaluate(method='textblob')
sentiment_analysis.predict_unlabeled_data(method='textblob')