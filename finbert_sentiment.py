import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

news_df = pd.read_csv('news_cleaned.csv', parse_dates=['Date'])

model_name = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

model.eval()

def get_finbert_sentiment_with_logits(text):
    if pd.isna(text) or text.strip() == "":
        return None, None, None, None
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze().tolist()  
    
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    neutral, positive, negative = probs
    
    compound_score = positive - negative
    
    return compound_score, positive, negative, logits

sentiments = news_df['cleaned_headline'].apply(get_finbert_sentiment_with_logits)
news_df['finbert_compound'] = sentiments.apply(lambda x: x[0] if x else None)
news_df['finbert_positive'] = sentiments.apply(lambda x: x[1] if x else None)
news_df['finbert_negative'] = sentiments.apply(lambda x: x[2] if x else None)
news_df['finbert_logits'] = sentiments.apply(lambda x: x[3] if x else None)

print("FinBERT sentiment analysis complete with raw logits.")
print(news_df[['cleaned_headline', 'finbert_compound', 'finbert_positive', 'finbert_negative', 'finbert_logits']].head())

news_df.to_csv('news_with_finbert_sentiment_and_logits.csv', index=False)
