# Modern Family (TV Show) Sentiment Analysis 

Project Description: In this project, I will develop a sentiment analysis tool to analyze Twitter data related to the popular TV show, Modern Family. The goal is to collect, process, and analyze tweets, comments, and posts from Twitter to determine the sentiment (positive, negative, or neutral) of the audience's reactions to the show. This project aims to provide valuable insights into how people perceive and engage with TV shows on social media platforms.

## Project Components:

### 1. Data Collection
Used Twitter scraping code to retreive tweets and comments from the official [@ModernFam](https://twitter.com/ModernFam) Twitter page. 
These comments were saved to json files which I combined into one. After removing duplicates, these are saved in the data folder. [This](data/twitter_comments.json) json file contains user, timestamp and text of the comment/tweet. There are approximate 1200 comments in this file, and these form the basis of the project.

### 2. Data Preprocessing:
Remove unnecessary data, including retweets and duplicates.
Clean the text data by handling special characters, and URLs.
Tokenize the text data to prepare it for sentiment analysis.
