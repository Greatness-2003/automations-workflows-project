# Modern Family (TV Show) Sentiment Analysis 

Project Description: In this project, I will develop a sentiment analysis tool to analyze Twitter data related to the popular TV show, Modern Family. The goal is to collect, process, and analyze tweets, comments, and posts from Twitter to determine the sentiment (positive, negative, or neutral) of the audience's reactions to the show. This project aims to provide valuable insights into how people perceive and engage with TV shows on social media platforms.

NB: For this project, I am only focusing on 1 TV show, but the process can be extended to others in the future.

Below are the various steps in the project. More may be added as it progresses.
## Project Components:

### 1. Data Collection (completed):
* Used Twitter scraping code to retreive tweets and comments from the official [@ModernFam](https://twitter.com/ModernFam) Twitter page. 
* These comments were saved to json files which I combined into one. After removing duplicates, these are saved in the data folder. 
* The json [file](data/twitter_comments.json) which contains user, timestamp and text of the comment/tweet is called `twiiter_comments`. 
* There are approximately 1300 comments in this file, and these form the basis of the project.


***NB: Tweets and comments can be collected using the Twitter Developer API, or by using Twitter scraping code. I used [pre-existing code](https://github.com/anwala/teaching-web-science/blob/main/fall-2023/week-3/twitter-scraper/scrape_twitter.py) developed by Professor Alexander Nwala, Data Science professor at William and Mary.***

### 2. Data Preprocessing (completed):
* I removed unnecessary data, including retweets and duplicates.
* Cleaned the text data by handling special characters, and URLs.
* Tokenized the text data to prepare it for sentiment analysis.
* This [preprocessing python class](preprocessing.py) found in `preprocessing.py` does all the above. 
* Running the class on `twiiter_comments` returns a [list](data/preprocessed_tweets.json) of tokenized and preprocessed tweets.

### 3. Sentiment Analysis (in progress):

* The class that performs sentiment analysis on `preprocessed_tweets.json` is called `SentimentAnalysis` and can be found [here](sentiment_analysis.py). 
* This class takes 2 approaches: TextBlob (rule-based approach) and Naive Bayes (machine learning approach). The user can choose which of the approaches they want to use when running the class.
* In order to train the naives bayes model, a random subset (600) of the comments were selected and manually labeled as positive, negative, or neutral. These labels are stored in another json [file](data/sentiment_labels.json) called `sentiment_labels`. 
* Evaluated the model's performance using metrics such as accuracy, precision, recall, and F1 score.
* TextBlob and the trained naive bayes model are then run on the unlabeled comments to predict sentiments. 

***NB: The random seed is set so that the same comments (whose labels are already stored) are retreived for training every time the class is run. If the user changes this value, or the number of comments to subset is changed, then they would have to do the manual labeling themselves.***


* The prediction function `predict_unlabeled_data` returns various statistics, such as percentages for negative, neutral and positive comments. For the data, the output for TextBlob method is below:

```
textblob statistics:
Total: 702
Positive: 365
Neutral: 282
Negative: 55
Positive Percentage: 51.99430199430199
Neutral Percentage: 40.17094017094017
Negative Percentage: 7.834757834757834
Yay, most people have a positive sentiment!
```

### 4. Data Visualization:

Create visualizations, such as word clouds, bar charts, and line graphs, to represent sentiment trends over time.
Display the distribution of positive, negative, and neutral tweets for different TV shows.
