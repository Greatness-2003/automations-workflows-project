import re
import json
import argparse

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:

    """
    A class for preprocessing textual data in JSON format.

    Attributes:
    - input_file (str): The input JSON file containing textual data.
    - output_file (str): The output JSON file to store preprocessed data.
    - data (list): The loaded JSON data.

    Methods:
    - decode_unicode_escape(text): Decodes Unicode-escaped characters in a text.
    - load_data(): Loads JSON data from the input file.
    - clean_unicode_escape(): Cleans Unicode-escaped characters in the loaded data.
    - clean_text(text): Removes URLs, tags, and newline characters from a text.
    - tokenize_and_filter(text): Tokenizes and filters alphanumeric words from a text.
    - remove_stopwords(tokens): Removes common English stopwords from a list of tokens.
    - preprocess(): Executes the entire preprocessing pipeline and writes the result to an output file.
    """


    def __init__(self, input_file, output_file):

        """
        Initializes the Preprocessor with input and output file names.

        Parameters:
        - input_file (str): The input JSON file containing textual data.
        - output_file (str): The output JSON file to store preprocessed data.
        """

        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def decode_unicode_escape(self, text):

        """
        Decodes Unicode-escaped characters in a text.

        Parameters:
        - text (str): The text with Unicode-escaped characters.

        Returns:
        - str: The text with decoded Unicode characters.
        """

        return text.encode().decode('unicode-escape')

    def load_data(self):

        """
        Loads JSON data from the input file into the 'data' attribute.
        """

        with open(self.input_file, 'r', encoding='utf-8') as infile:
            self.data = json.load(infile)

    def clean_unicode_escape(self):

        """
        Cleans Unicode-escaped characters in the loaded data.
        """

        for item in self.data:
            item['content'] = self.decode_unicode_escape(item['content'])
            item['username'] = self.decode_unicode_escape(item['username'])

    def clean_text(self, text):

        """
        Removes URLs, tags, and newline characters from a text.

        Parameters:
        - text (str): The input text.

        Returns:
        - str: The text with URLs, tags, and newlines removed.
        """

        text = re.sub(r'http\S+|www\S+|https\S+|\n', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    def tokenize_and_filter(self, text):

        """
        Tokenizes and filters alphanumeric words from a text.

        Parameters:
        - text (str): The input text.

        Returns:
        - list: The list of filtered tokens.
        """

        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        return tokens

    def remove_stopwords(self, tokens):

        """
        Removes common English stopwords from a list of tokens.

        Parameters:
        - tokens (list): The list of tokens.

        Returns:
        - list: The list of tokens with stopwords removed.
        """

        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word not in stop_words]

    def preprocess(self):

        """
        Executes the entire preprocessing pipeline and writes the result to the output file.
        """

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


def main():

    """
    The main function that parses command-line arguments and executes the preprocessing.
    """
    
    parser = argparse.ArgumentParser(description='Text Preprocessing Script')
    parser.add_argument('input_file', help='Input JSON file name')
    parser.add_argument('output_file', help='Output JSON file name')
    
    args = parser.parse_args()

    preprocessor = Preprocessor(input_file=args.input_file, output_file=args.output_file)
    preprocessor.preprocess()

if __name__ == "__main__":
    main()