import argparse
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sentiment_analysis import SentimentAnalysis


class SentimentVisualizer:
    def __init__(self, sentiment_analysis):
        self.sentiment_analysis = sentiment_analysis

    def sentiment_distribution_bar(self, method_stats, method):
        # create and display a bar chart representing the distribution of sentiments for a given method

        labels = ['Negative', 'Neutral', 'Positive']
        percentages = [method_stats['Negative Percentage'], method_stats['Neutral Percentage'], method_stats['Positive Percentage']]

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.bar(labels, percentages, color=['red', 'gray', 'green'])

        ax.set_ylabel('Percentage', fontsize=19, fontfamily='cambria', fontstyle='oblique', fontweight='semibold')
        ax.set_title(f'Sentiment Distribution ({method})', fontsize=19, fontfamily='cambria', fontstyle='oblique', fontweight='semibold')

        ax.set_xticklabels(labels, fontsize=15, fontfamily='cambria')
        ax.set_yticklabels([f'{tick}%' for tick in ax.get_yticks()], fontsize=14, fontfamily='cambria')
        plt.show()

    def sentiment_distribution_stackedbar(self, method_stats_list):
        # create and display a stacked bar chart comparing sentiment distribution across different methods

        methods = [method['Method'] for method in method_stats_list]
        positive_percentages = [method['Positive Percentage'] for method in method_stats_list]
        neutral_percentages = [method['Neutral Percentage'] for method in method_stats_list]
        negative_percentages = [method['Negative Percentage'] for method in method_stats_list]

        bar_width = 0.35
        index = np.arange(len(methods))

        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(index, positive_percentages, bar_width, label='Positive')
        bars2 = ax.bar(index, neutral_percentages, bar_width, label='Neutral', bottom=positive_percentages)
        bars3 = ax.bar(index, negative_percentages, bar_width, label='Negative', bottom=np.array(neutral_percentages) + np.array(positive_percentages))

        ax.set_xlabel('Methods', fontsize=19, fontfamily='cambria', fontstyle='oblique', fontweight='semibold')
        ax.set_ylabel('Percentage', fontsize=19, fontfamily='cambria', fontstyle='oblique', fontweight='semibold')
        ax.set_title('Sentiment Distribution Comparison', fontsize=19, fontfamily='cambria', fontstyle='oblique', fontweight='semibold')
        ax.set_xticks(index)
        ax.set_xticklabels(methods, fontsize=15, fontfamily='cambria')
        ax.set_yticklabels([f'{tick}%' for tick in ax.get_yticks()], fontsize=15, fontfamily='cambria')

        legend = ax.legend()
        legend.set_title('Sentiment', prop={'size': 16, 'weight': 'semibold', 'family': 'cambria'})
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_family('cambria')

        plt.show()

    def generate_word_cloud(self, method='naive_bayes'):
        # generate and display a word cloud for each sentiment category

        sentiment_labels = ['Negative', 'Neutral', 'Positive']

        for sentiment in sentiment_labels:
            if method == 'naive_bayes':
                sentiment_tweets = [tweet for tweet in self.sentiment_analysis.unlabeled_tweets if
                                    self.sentiment_analysis.analyze_sentiment_naive_bayes(tweet) == sentiment_labels.index(sentiment)]
            elif method == 'textblob':
                sentiment_tweets = [tweet for tweet in self.sentiment_analysis.unlabeled_tweets if
                                    self.sentiment_analysis.analyze_sentiment_textblob(tweet) == sentiment_labels.index(sentiment)]
            else:
                raise ValueError(f"Invalid method: {method}. Choose either 'naive_bayes' or 'textblob'.")

            if sentiment_tweets:
                sentiment_text = ' '.join(sentiment_tweets)

                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Word Cloud for {sentiment} Sentiment')
                plt.axis('off')
                plt.show()
            else:
                print(f"No comments for {sentiment} sentiment.")

def main():
    parser = argparse.ArgumentParser(description='Sentiment Visualizer Script')
    parser.add_argument('--method', choices=['naive_bayes', 'textblob'], default='naive_bayes',
                        help='Method for sentiment analysis (naive_bayes or textblob)')

    args = parser.parse_args()

    sentiment_analysis = SentimentAnalysis()
    visualizer = SentimentVisualizer(sentiment_analysis, method=args.method)
    sentiment_analysis.train_classifier(method=args.method)

    method_stats = sentiment_analysis.predict_unlabeled_data(method=args.method)
    visualizer.sentiment_distribution_bar(method_stats)

    method_stats_textblob = sentiment_analysis.predict_unlabeled_data(method='textblob')
    visualizer.sentiment_distribution_bar(method_stats_textblob)

    method_stats_list = [
        {'Method': 'Naive Bayes', **method_stats},
        {'Method': 'TextBlob', **method_stats_textblob}
    ]

    visualizer.sentiment_distribution_stackedbar(method_stats_list)
    visualizer.generate_word_cloud()

if __name__ == "__main__":
    main()
