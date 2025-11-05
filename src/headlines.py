from transformers import pipeline
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from newsapi import NewsApiClient

#headlines extractions and sentiment classification

load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

