# Sentiment Analysis on Reddit Comments

## Overview
This project is aimed at performing sentiment analysis on Reddit comments using various natural language processing (NLP) techniques. Social media platforms provide a rich source of unstructured text
data. The data can be converted into valuable information by using sentiment analysis. This project will classify social media post’s comment’s sentiments as Positive (POS), Negative (NEG),
or Neutral (NEU).

The workflow includes extracting Reddit comments using the Reddit API via the Python PRAW library, annotating the comments using a pre-trained model from Hugging Face, comparing three different feature extraction methods (Bag of Words, TF-IDF, BERT embeddings), and training a Support Vector Machine (SVM) classifier to classify comments into three categories: positive, negative, and neutral. Additionally, a graphical user interface (GUI) is developed using Python Tkinter for ease of use.

## METHODOLOGY:
### 1. DATA COLLECTION
##### Reddit Comments Extraction:
      * Account Creation: Created a Reddit account to access the Reddit API.
      * App Creation: Created a Reddit App to obtain API keys and credentials.
      * Access Reddit API: The team chose a subreddit called “Canada” community(https://www.reddit.com/r/canada/) and extracted comments for 50 posts (nearly 2700 comments) initially. 
      * Then 100 posts later (nearly 7000 comments)
      * Utilized the PRAW (Python Reddit API Wrapper) to extract comments. 
      * Extracted comments are saved: The extracted comments are put into a JSON file for further processing.
##### DATA LOADING
      * The data is loaded into a Pandas DataFrame
##### ANNOTATE DATASET
      * To annotate the comments, a pre-trained BERTweet sentiment analysis model is employed.
      * Reference: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
      * The model is loaded using the Hugging Face Transformers library.
      * This uses the labels POS, NEG, NEU for sentiment analysis.
            * To ensure optimal compatibility with the sentiment analysis model, comments are filtered based on their length.
            * Comments outside the specified range (between min_sequence_length and max_sequence_length) are excluded from further analysis.
            * The sentiment analysis model is applied to each filtered comment, and the predicted sentiment label is stored in a new column, 'predicted_sentiment'.
##### DATA PREPROCESSING:
    * Following sentiment annotation, the comments data undergoes preprocessing steps to enhance the quality of the text data. This includes tokenization, stop-word removal,lowercasing, and handling special              characters.
    * Tried to address majority of challenges specific to social media text (hashtags, emojis,
    slang).
##### Feature Extraction:
    * Bag-Of-Words
    * TF-IDF
    * BERT
##### MODEL:
    * Model: SVM
## Insights
The sentiment analysis project provided valuable insights into 
    * The complexities of natural language processing(NLP)
    * Emphasizing the need for adaptability, 
    * Context-aware analysis, and 
    * Continuous improvement.

