# Modern Family (TV Show) Sentiment Analysis 

Project Description: In this project, I will develop a sentiment analysis tool to analyze Twitter data related to the popular TV show, Modern Family. The goal is to collect, process, and analyze tweets, comments, and posts from Twitter to determine the sentiment (positive, negative, or neutral) of the audience's reactions to the show. This project aims to provide valuable insights into how people perceive and engage with TV shows on social media platforms.

NB: For this project, I am only focusing on 1 TV show, but the process can be extended to others in the future.

Below are the various steps in the project. More may be added as it progresses.
## Project Components:

### 1. Data Collection (completed):
* Used Twitter scraping code to retreive tweets and comments from the official [@ModernFam](https://twitter.com/ModernFam) Twitter page. 
* These comments were saved to json files which I combined into one. After removing duplicates, these are saved in the data folder. 
* [This](data/twitter_comments.json) json file contains user, timestamp and text of the comment/tweet. 
* There are approximately 1300 comments in this file, and these form the basis of the project.

*NB: Tweets and comments can be collected using the Twitter Developer API, or by using Twitter scraping code. I used [pre-existing code](https://github.com/anwala/teaching-web-science/blob/main/fall-2023/week-3/twitter-scraper/scrape_twitter.py) developed by Professor Alexander Nwala, Data Science professor at William and Mary.*

### 2. Data Preprocessing (completed):
* I removed unnecessary data, including retweets and duplicates.
* Cleaned the text data by handling special characters, and URLs.
* Tokenized the text data to prepare it for sentiment analysis.
* This [preprocessing python class](preprocessing.py) does all the above. 

### 3. Sentiment Analysis (in progress):

I used two approaches:
* Rule-Based Approaches: Used pre-built sentiment analysis library, TextBlob to perform sentiment analysis.
* Machine Learning Approaches: Trained own sentiment analysis model (Naive Bayes) using labeled data.
* Implemented a sentiment analysis algorithm to classify tweets into three categories: positive, negative, or neutral.
* Evaluated the model's performance using metrics such as accuracy, precision, recall, and F1 score.
* Found that TextBlob gave better performance metrics, so went with that.


### 4. Data Visualization:

Create visualizations, such as word clouds, bar charts, and line graphs, to represent sentiment trends over time.
Display the distribution of positive, negative, and neutral tweets for different TV shows.
