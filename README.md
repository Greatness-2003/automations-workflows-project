# Modern Family (TV Show) Sentiment Analysis 

Project Description: In this project, I will develop a sentiment analysis tool to analyze Twitter data related to the popular TV show, Modern Family. The goal is to collect, process, and analyze tweets, comments, and posts from Twitter to determine the sentiment (positive, negative, or neutral) of the audience's reactions to the show. This project aims to provide valuable insights into how people perceive and engage with TV shows on social media platforms.

NB: For this project, I am only focusing on 1 TV show, but the process can be extended to others in the future.

Below are the various steps in the project. More may be added as it progresses.
## Project Components:

### 1. Data Collection (completed):
* Used Twitter scraping code to retreive tweets and comments from the official [@ModernFam](https://twitter.com/ModernFam) Twitter page. 
* These comments were saved to json files which I combined into one. After removing duplicates, these are saved in the data folder. 
* [This](data/twitter_comments.json) json file contains user, timestamp and text of the comment/tweet. 
* There are approximate 1200 comments in this file, and these form the basis of the project.

### 2. Data Preprocessing (In progress):
* Remove unnecessary data, including retweets and duplicates.
* Clean the text data by handling special characters, and URLs.
* Tokenize the text data to prepare it for sentiment analysis.

### 3. Sentiment Analysis:

* Implement a sentiment analysis algorithm to classify tweets into three categories: positive, negative, or neutral.
* Utilize natural language processing (NLP) libraries like NLTK for text analysis.
* Train a machine learning model (e.g., Naive Bayes, LSTM, or BERT) to perform sentiment classification.
* Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score.

### 4. Data Visualization:

Create visualizations, such as word clouds, bar charts, and line graphs, to represent sentiment trends over time.
Display the distribution of positive, negative, and neutral tweets for different TV shows.
