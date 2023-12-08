# Modern Family (TV Show) Sentiment Analysis 

Project Description: In this project, I will develop a sentiment analysis tool to analyze Twitter data related to the popular TV show, Modern Family. The goal is to collect, process, and analyze tweets, comments, and posts from Twitter to determine the sentiment (positive, negative, or neutral) of the audience's reactions to the show. This project aims to provide valuable insights into how people perceive and engage with TV shows on social media platforms.

***NB: For this project, I am only focusing on 1 TV show, but the process can be extended to others in the future.***

Below are the various steps in the project. More may be added as it progresses.
## Project Components:

### 1. Data Collection (completed):
* Used Twitter scraping code to retreive tweets and comments from the official [@ModernFam](https://twitter.com/ModernFam) Twitter page. 
* These comments were saved to json files which I combined into one. After removing duplicates, these are saved in the data folder. 
* The json [file](data/twitter_comments.json) which contains user, timestamp and text of the comment/tweet is called `twiiter_comments`. 
* There are approximately 1300 comments in this file, and these form the basis of the project.
* These comments were collected mainly for training and testing purposes. The user would have to should collect their data themselves to run the program on.


***NB: Tweets and comments can be collected using the Twitter Developer API, or by using Twitter scraping code. I used [pre-existing code](https://github.com/anwala/teaching-web-science/blob/main/fall-2023/week-3/twitter-scraper/scrape_twitter.py) developed by Professor Alexander Nwala, Data Science professor at William and Mary.***

### 2. Data Preprocessing (completed):
* I removed unnecessary data, including retweets and duplicates.
* Cleaned the text data by handling special characters, and URLs.
* Tokenized the text data to prepare it for sentiment analysis.
* This [preprocessing python class](preprocessing.py) found in `preprocessing.py` does all the above. 
* Running the class on `twiiter_comments` returns a [json file](data/preprocessed_tweets.json) of tokenized and preprocessed tweets.
* The user should run the class on their collected comments and save the output as `unlabeled_comments.json` in the `data` folder. This is necessary to run the sentiment analysis.

Example terminal run:

```bash
python preprocessing.py data/input_data.json data/output_data.json
```
### 3. Sentiment Analysis (completed):

* The class that performs sentiment analysis on `preprocessed_tweets.json` is called `SentimentAnalysis` and can be found [here](sentiment_analysis.py). 
* This class takes 2 approaches: TextBlob (rule-based approach) and Naive Bayes (machine learning approach). The user can choose which of the approaches they want to use when running the script. The default method is `naive_bayes`, and will be implemented if the user does not give an argument.

```bash
python sentiment_analysis.py --method textblob
```

* In order to train the naives bayes model, a random subset (600) of the comments were selected and manually labeled as positive, negative, or neutral. These labels are stored in another json [file](data/sentiment_labels.json) called `sentiment_labels`. 
* Evaluated the model's performance using metrics such as accuracy, precision, recall, and F1 score.
* TextBlob and the trained naive bayes model are then run on the unlabeled comments to predict sentiments. 
* If the user's preprocessed comments (`unlabeled_comments.json`) do not exist, then the program would run prediction on the remaining 700 comments that weren't labeled.


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
Yay, most people love this show!
```

### 4. Data Visualization (completed):

* Wrote a Class that create several visualizations, called `SentimentVisualizer` located in this [file](analysis_plots.py). This class takes the `SentimentAnalysis` class as a parameter and uses some of its functions.
* First function creates a bar chart to show the distribution of sentiments in the unlabeled dataset. This helps visualize the balance or imbalance between positive, neutral, and negative sentiments.
* Next function creates a stacked bar chart to compare the distribution of sentiments predicted by different methods ( Naive Bayes vs. TextBlob). This can help identify patterns and differences between the methods.
* Last function generate word clouds for each sentiment category to highlight the most frequent words associated with positive, neutral, and negative sentiments. This can offer insights into the key terms driving sentiment.

Example terminal run:

```bash
python analysis_plots.py --method textblob
```

* Just like the previous, the default method is `naive_bayes`.