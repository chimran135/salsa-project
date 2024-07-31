from SentimentAnalysis import *
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.concurrency import run_in_threadpool
import shutil
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load Application Configurations using "python-dotenv"
from dotenv import dotenv_values

import time
import re
# from tqdm.notebook import tqdm
# tqdm.pandas()

# import os
# import logging
#logging.basicConfig(level=logging.INFO)
import logging as log
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(dir_path+"/logs"):
    os.makedirs(dir_path+"/logs")

log.basicConfig(level=log.DEBUG,
                format='%(asctime)s: %(levelname)s [%(filename)s:%(lineno)s] %(message)s',
                datefmt='%d/%m/%Y %I:%M:%S %p',
                handlers=[log.FileHandler(dir_path+'/logs/salsa-logs.log'),
                          log.StreamHandler()]
                )

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests
    allow_headers=["*"],  # Allow all headers
)

class SentimentAnalysisWraper:  
    data_processing = DataPreprocessing()
    english_sentiment_analysis = EnglishSentimentAnalysis()
    spanish_sentiment_analysis = SpanishSentimentAnalysis()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        pass

    @staticmethod
    def get_sentiment(data, lang='en'):
        
        # Access Data Preprocessing class attributes and methods
        data_processing = SentimentAnalysisWraper.data_processing

        # Select Sentiment Analizer based on Language
        if(lang == 'en'):
            sentiment_analysis = SentimentAnalysisWraper.english_sentiment_analysis
        elif(lang == 'es'):
            sentiment_analysis = SentimentAnalysisWraper.spanish_sentiment_analysis

        # Load the configuration from .env file
        config = dotenv_values("config.env")

        # Perform data preprocessing
        data = data_processing.remove_words_and_emojis(data)
        data = data_processing.remove_emojis(data)
        data = data_processing.fix_punctuation(data)

        # Convert Single Instance String data into a list for Batch SpaCy
        data = [data]
        
        # Lemmatize the data using SpaCy
        #batch_spacy = data_processing.get_batch_spacy(data, lang)
        
        # Only for model
        if(config['PARSER_TYPE'] == 'MODEL'):
            # Lemmatize the data using SpaCy
            #batch_spacy = data_processing.get_batch_spacy(data, lang)
            split_review = True if config['SPLIT_REVIEW'] == 'True' else False
            batch_spacy = data_processing.get_batch_spacy(data, lang, split_review)
            #batch_spacy = data_processing.get_batch_spacy(data, lang)

        #score = sentiment_analysis.calcSentenceScore(data, batch_spacy[0], lang)

        if(config['SENTIMENT_CALCULATION_MODE'] == 'MAXIMA'):
            if(config['PARSER_TYPE'] == 'MODEL'):
                score = sentiment_analysis.calcSentenceScoreMaxima(data, batch_spacy[0], lang)
                score = sentiment_analysis.calcSentenceScoreMaxima(data, batch_spacy, lang)
            elif(config['PARSER_TYPE'] == 'STANZA'):
                score = sentiment_analysis.calcSentenceScoreMaxima(data)
        else:
            if(config['PARSER_TYPE'] == 'MODEL'):
                #score = sentiment_analysis.calcSentenceScore(data, batch_spacy[0], lang)
                #score = sentiment_analysis.calcReviewScore(data, batch_spacy[0], lang) # 31.5
                score = sentiment_analysis.calcReviewScore(data, batch_spacy, lang) # 02.06
            elif(config['PARSER_TYPE'] == 'STANZA'):
                #score = sentiment_analysis.calcSentenceScore(data)
                score = sentiment_analysis.calcReviewScore(data)

        conllu_dec_trees = sentiment_analysis.dec_tree
        conllu_data = []
        for conllu_dec_tree in conllu_dec_trees:
            conllu_tree_data = sentiment_analysis.modify_conllu_data(conllu_dec_tree, score["lstTokens"], score["lstScores"], score["lstModifiers"])
            conllu_data.append(conllu_tree_data)

        #conllu_data = sentiment_analysis.modify_conllu_data(sentiment_analysis.dec_tree, score["lstTokens"], score["lstScores"], score["lstModifiers"])

        normalized_score = sentiment_analysis.normalize_value(score["score"])
        
        # Check for normalize range [-5, 5] to [1, 5]
        if (normalized_score > 5):
            normalized_score = 5
        elif(normalized_score < -5):
            normalized_score = 1
        polarity_label = sentiment_analysis.system_polarity_label(normalized_score)

        return [polarity_label, normalized_score, conllu_data]

    @staticmethod
    def get_file_sentiment(file_path, lang='en'):
        # Access class attributes using class name
        data_processing = SentimentAnalysisWraper.data_processing

        if(lang == 'en'):
            sentiment_analysis = SentimentAnalysisWraper.english_sentiment_analysis
        elif(lang == 'es'):
            sentiment_analysis = SentimentAnalysisWraper.spanish_sentiment_analysis

        # Preprocessing Steps are handled in "get_data()" function
        data = data_processing.get_data(file_path)
        #data['reviewCorr'] = data['reviewCorr'].progress_apply(sentiment_analysis.split_into_sentences)

        # Load the configuration from .env file
        config = dotenv_values("config.env")

        # Only for model
        if(config['PARSER_TYPE'] == 'MODEL'):
            # Lemmatize the data using SpaCy
            #batch_spacy = data_processing.get_batch_spacy(data['reviewCorr'].tolist(), lang)
            split_review = True if config['SPLIT_REVIEW'] == 'True' else False
            batch_spacy = data_processing.get_batch_spacy(data['reviewCorr'].tolist(), lang, split_review)
        
        scores = []
        summary = {}
        t0 = time.time()
        #for i, row in tqdm(data.iterrows()):
        for i, row in data.iterrows():
            #print(i, row.reviewCorr, '\n')
            if(config['SENTIMENT_CALCULATION_MODE'] == 'MAXIMA'):
                if(config['PARSER_TYPE'] == 'MODEL'):
                    score = sentiment_analysis.calcSentenceScoreMaxima(row.reviewCorr, batch_spacy[i], lang)
                elif(config['PARSER_TYPE'] == 'STANZA'):
                    score = sentiment_analysis.calcSentenceScoreMaxima(row.reviewCorr)
            else:
                if(config['PARSER_TYPE'] == 'MODEL'):
                    #score = sentiment_analysis.calcSentenceScore(row.reviewCorr, batch_spacy[i], lang)
                    score = sentiment_analysis.calcReviewScore(row.reviewCorr, batch_spacy[i], lang)
                elif(config['PARSER_TYPE'] == 'STANZA'):
                    #score = sentiment_analysis.calcSentenceScore(row.reviewCorr)
                    score = sentiment_analysis.calcReviewScore(row.reviewCorr)

            #conllu_data = sentiment_analysis.modify_conllu_data(sentiment_analysis.dec_tree, score["lstTokens"], score["lstScores"], score["lstModifiers"])
            conllu_dec_trees = sentiment_analysis.dec_tree

            conllu_data = []
            for conllu_dec_tree in conllu_dec_trees:
                conllu_tree_data = sentiment_analysis.modify_conllu_data(conllu_dec_tree, score["lstTokens"], score["lstScores"], score["lstModifiers"])
                conllu_data.append(conllu_tree_data)
            
            # Check if the system failed to predict sentimnet score
            if(score["score"] in [-100, -200, -300, 'none']):
                normalized_score = ''
            else:
                normalized_score = sentiment_analysis.normalize_value(score["score"])
                # Check for normalize range [-5, 5] to [1, 5]
                if (normalized_score > 5):
                    normalized_score = 5
                elif(normalized_score < -5):
                    normalized_score = 1
            polarity_label = sentiment_analysis.system_polarity_label(normalized_score)
            polarity_label = polarity_label.strip()

            # prepare data for Pie Chart
            if(polarity_label not in ["Positive", "Negative", "Neutral"]):
                #polarity_label = 'Positive' # Default "Positive"
                polarity_label = sentiment_analysis.get_default_polarity_label() # Default "Positive"
            # If the key exists in the dictionary, increment its value by 1
            if polarity_label in summary:
                summary[polarity_label] += 1
            # If the key does not exist in the dictionary, add it with a value of 1
            else:
                summary[polarity_label] = 1
            
            scores.append({'id': i, 'text': row.reviewCorr, 'sentiment': polarity_label, 'polarity': normalized_score, 'conll': conllu_data})
        
        t1 = time.time()
        total = t1-t0
        #print('Total Time (Seconds): ' + str(total))

        summary = dict(sorted(summary.items(), key=lambda item: item[0]))
        #print(summary)
        # Get the list of keys
        labels = list(summary.keys())
        # Get the list of values
        values = list(summary.values())
        return {"scores": scores,  "summary": {"labels": labels, "values": values}}
        #return scores

@app.post("/submit")
async def submit_form(data: dict):
    lang = data.get("lang")
    text = data.get("text")
    if lang is None or text is None:
        raise HTTPException(status_code=422, detail="Language and Text are required.")
    
    # Process the received data
    sentiment = SentimentAnalysisWraper.get_sentiment(text, lang)
    processed_data = {'lang': lang, 'text': text, 'polarity': sentiment[1], 'sentiment': sentiment[0], 'conll': sentiment[2]}
    
    # Return the processed data as a response
    return {"message": processed_data}

# @app.post("/upload/")
# async def upload_file(lang: str = Form(...), file: UploadFile = File(...)):
#     """
#     This endpoint receives a file and a selected language value from a dropdown field, and saves the file to the server.
#     """
#     file_path = f"uploaded_files/{file.filename}"

#     # Save the file to a specified location
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     # Perform Sentiment Analysis on the received file
#     processed_data = SentimentAnalysisWraper.get_file_sentiment(file_path, lang)
#     return {"message": processed_data}

# @app.post("/upload/")
# async def upload_file(lang: str = Form(...), file: UploadFile = File(...)):
#     file_path = f"uploaded_files/{file.filename}"
    
#     def save_file():
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#     # Save the file to a specified location
#     await run_in_threadpool(save_file)

#     # Perform Sentiment Analysis on the received file
#     processed_data = SentimentAnalysisWraper.get_file_sentiment(file_path, lang)
#     return {"message": processed_data}

@app.post("/upload/")
async def upload_file(lang: str = Form(...), file: UploadFile = File(...)):
    file_path = f"uploaded_files/{file.filename}"
    
    # Log file path
    log.info(f"File path: {file_path}")
    
    # Save the file to a specified location
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        log.info(f"File saved successfully at {file_path}")
    except Exception as e:
        log.error(f"Failed to save file: {e}")
        return {"error": "Failed to save file"}

    # Perform Sentiment Analysis on the received file
    try:
        processed_data = SentimentAnalysisWraper.get_file_sentiment(file_path, lang)
        log.info(f"Processed data: {processed_data}")
    except Exception as e:
        log.error(f"Failed to process file: {e}")
        return {"error": "Failed to process file"}
    
    return {"message": processed_data}