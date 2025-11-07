from transformers import pipeline
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import requests

#headlines extractions and sentiment classification

load_dotenv()
gnews_api_key = os.getenv("GNEWS_API_KEY")
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

#Keywords related to inflation used to identify relevant articles
keywords = ["cpi", "inflation", "consumer price", "pce", "price index", "ppi",
            "cost of living", "price pressure", "rate hike", "monetary policy",
            "fed", "federal reserve", "interest rate", "disinflation", "deflation"]

#TODO: do for gnews
def get_headlines(sources:list[str],keywords:list[str],from_date:str,to_date:str)->list[str]:
    keywords_string = ",".join(keywords)
    print(keywords_string)
    sources_string = ",".join(sources)
    print(sources_string)
    response = api.get_everything(qintitle=keywords_string,sources=sources_string, from_param=from_date, to=to_date, language="en")
    headlines = []
    for article in response["articles"]:
        headlines.append(article["title"])
    return headlines
                  
                  
                  


