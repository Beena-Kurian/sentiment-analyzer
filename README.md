# SENTIMENT ANALYSIS ON REDDIT COMMENTS 
Social media platforms generate a massive amount of textual data with diverse sentiments. The objective of this project is to develop a sentiment analysis system capable of automatically classifying social media posts into positive, negative, or neutral sentiments.

## OVERVIEW
This project is aimed at performing sentiment analysis on Reddit comments using various natural language processing (NLP) techniques. Social media platforms provide a rich source of unstructured text.
data. The data can be converted into valuable information by using sentiment analysis. This project will classify social media post’s comment’s sentiments as Positive (POS), Negative (NEG),
or Neutral (NEU).

The workflow includes extracting Reddit comments using the Reddit API via the Python PRAW library, annotating the comments using a pre-trained model from Hugging Face, comparing three different feature extraction methods (Bag of Words, TF-IDF, BERT embeddings), and training a Support Vector Machine (SVM) classifier to classify comments into three categories: positive, negative, and neutral. Additionally, a graphical user interface (GUI) is developed using Python Tkinter for ease of use.

## METHODOLOGY:
### 1. DATA COLLECTION
##### Reddit Comments Extraction:
* Account Creation: Created a Reddit account to access the Reddit API.
* App Creation: Created a Reddit App to obtain API keys and credentials.
* Access Reddit API: taken subreddit called “Canada” community(https://www.reddit.com/r/canada/) and extracted comments for 100 posts (nearly 7000 comments) 
* Utilized the PRAW (Python Reddit API Wrapper) to extract comments. 
* Extracted comments are saved: The extracted comments are put into a JSON file for further processing.
##### DATA LOADING
The data is loaded into a Pandas DataFrame
##### ANNOTATE DATASET
To annotate the comments, a pre-trained BERTweet sentiment analysis model is employed.
Reference: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
The model is loaded using the Hugging Face Transformers library.
This uses the labels POS, NEG, NEU for sentiment analysis.
To ensure optimal compatibility with the sentiment analysis model, comments are filtered based on their length.
Comments outside the specified range (between min_sequence_length and max_sequence_length) are excluded from further analysis.
The sentiment analysis model is applied to each filtered comment, and the predicted sentiment label is stored in a new column, 'predicted_sentiment'.
##### DATA PREPROCESSING:
Following sentiment annotation, the comments data undergoes preprocessing steps to enhance the quality of the text data. This includes tokenization, stop-word removal,lowercasing, and handling special             characters.
Tried to address majority of challenges specific to social media text (hashtags, emojis, slang).
##### Feature Extraction:
* Bag-Of-Words
* TF-IDF
* BERT
##### MODEL:
* Model: SVM
## Insights
The sentiment analysis project provided valuable insights into the complexities of natural language processing(NLP),emphasizing the need for adaptability,context-aware analysis, and continuous improvement.

## Usage examples
* Initially create reddit account and app, then utilized the PRAW (for this execute `collect_reddit_data.ipynb` with your reddit account credentials)
* After successful execution you can find `reddit_comments_data.json` file, which contain extracted comments with their score.
* Then to annotate the extracted comments 'annotate_comments.ipynb`, after annotations you will get, `final_dataset.csv`.In contain columns with comment body, score, comment length, and predicted_sentiment
* Then performed the preprocessing by executing `preprocessing.ipynb`. I have saved a copy of the dataset as `before_preprocessing.csv`.
* After preprocessing dataset became `preprocessed_output.csv`, where you can find the tokens,stemmed_tokens, and lemmatized_tokens along with comment body,score and comment_length.
* You can check the csv files to learn the difference after each step.
* Now you have the the annotated and preprocessed dataset with you.
* Next you can execute any of the 3 files
  - `SVM_BOW_feature_extraction_model_build.ipnb`
  - `SVM_tfidf_feature_extraction_model_build.ipnb`
  - `SVM_BERT_feature_extraction_model_build.ipnb`
# Results

## Model Performance
-----------------------------------------------------------------------
| Feature Extraction | Model | Training Accuracy  | Testing Accuracy  |
| ------------------ | ----- | ------------------ | ----------------- |
| BOW                | SVM   | 0.8147             | 0.6491            |
| TF-IDF             | SVM   | 0.8737             | 0.6556            |
| BERT               | SVM   | 0.8520             | 0.7337            |
-----------------------------------------------------------------------


