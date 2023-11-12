import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def decode_unicode_escape(self, text):
        return text.encode().decode('unicode-escape')

    def load_data(self):
        with open(self.input_file, 'r', encoding='utf-8') as infile:
            self.data = json.load(infile)

    def clean_unicode_escape(self):
        for item in self.data:
            item['content'] = self.decode_unicode_escape(item['content'])
            item['username'] = self.decode_unicode_escape(item['username'])

    def clean_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+|\n', '', text)
        # Remove tags
        text = re.sub(r'@\w+', '', text)
        return text

    def tokenize_and_filter(self, text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        return tokens

    def remove_stopwords(self, tokens):
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]

    def preprocess(self):
        self.load_data()
        self.clean_unicode_escape()

        cleaned_tweets = []
        for tweet in self.data:
            text = self.clean_text(tweet['content'])
            tokens = self.tokenize_and_filter(text)
            tokens = self.remove_stopwords(tokens)
            cleaned_tweets.append(tokens)

        preprocessed_tweets = [', '.join(word for word in tweet) for tweet in cleaned_tweets]
        # remove empty strings
        preprocessed_tweets = [lst for lst in preprocessed_tweets if not all(word.strip() == '' for word in lst)]

        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            json.dump(preprocessed_tweets, outfile, ensure_ascii=False, indent=1)

        print(f"Preprocessed data written to {self.output_file}")

# Example usage
input_file = 'twitter_comments.json'
output_file = 'preprocessed_tweets.json'

preprocessor = Preprocessor(input_file, output_file)
preprocessor.preprocess()
