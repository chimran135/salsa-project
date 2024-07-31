from utilities import *
import transformers
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from src.models.deps_tree import D_Tree
from src.models.deps_label import D_Label
from src.utils.constants import D_ROOT_HEAD, D_NULLHEAD, D_ROOT_REL, D_POSROOT, D_EMPTYREL, D_2P_GREED, D_2P_PROP
from src.encs.enc_deps import *
from src.utils.constants import *
from src.models.linearized_tree import LinearizedTree

import io
import os
import sys
import time
import string
import re
import numpy as np
import pandas as pd

from pickle import TRUE

# SpaCy
import spacy
from spacy.language import Language
spacy.prefer_gpu

# Stanza
import stanza

# NLTK
import nltk
from nltk.tokenize import sent_tokenize
# Download the necessary NLTK resources
nltk = nltk.download('punkt')

# Compute the Sentence Score
from statistics import mean

# Load Application Configurations using "python-dotenv"
from dotenv import dotenv_values

class DataPreprocessing:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(cls, "Object Created!")
        return cls._instance
    
    def __init__(self):
        pass
    
    def get_file_extension(self, file_path):
        """
        Get the file extension from the file path.

        Parameters:
            file_path (str): The path of the file.

        Returns:
            str: The file extension (including the dot).
        """
        return file_path[file_path.rfind('.'):].lower()

    def sentence_case(self, text):
        # Split into sentences. Therefore, find all text that ends
        # with punctuation followed by white space or end of string.
        sentences = re.findall(r'(?:\d+\.\d+|\b[A-Z](?:\.[A-Z])*\b\.?|[^.!?])+[.!?](?:\s|\Z)', text)

        # Capitalize the first letter of each sentence
        sentences = [x[0].upper() + x[1:] for x in sentences]
        #print(sentences)
        # Combine sentences
        return ''.join(sentences)

    def fix_punctuation(self, text):
        #lower case
        try:
            text = text.lower()
            #txt1 = sentence_case(text)
            #add space after punctuation
            text1 = re.sub(r'(\d+\.\d+|\b[A-Z](?:\.[A-Z])*\b\.?)|([.,;:!?)])\s*', lambda x: x.group(1) or f'{x.group(2)} ', text)
            return text1
        except:
            return text

    # Remove Emojis From Text
    def remove_words_and_emojis(self, text):
        # Sample list of words/emojis to remove
        words_to_remove = ['$:', '%)', '%-)', '&-:', '&:', "( '}{' )", '(%', "('-:", "(':", '((-:', '(*', '(-%', '(-*', '(-:', '(-:0', '(-:<', '(-:o', '(-:O', '(-:{', '(-:|>*', '(-;', '(-;|', '(8', '(:', '(:0', '(:<', '(:o', '(:O', '(;', '(;<', '(=', '(?:', '(^:', '(^;', '(^;0', '(^;o', '(o:', ")':", ")-':", ')-:', ')-:<', ')-:{', '):', '):<', '):{', ');<', '*)', '*-)', '*-:', '*-;', '*:', '*<|:-)', '*\\0/*', '*^:', ',-:', "---'-;-{@", '--<--<@', '.-:', '..###-:', '..###:', '/-:', '/:', '/:<', '/=', '/^:', '/o:', '0-8', '0-|', '0:)', '0:-)', '0:-3', '0:03', '0;^)', '0_o', '10q', '1337', '143', '1432', '14aa41', '182', '187', '2g2b4g', '2g2bt', '2qt', '3:(', '3:)', '3:-(', '3:-)', '4col', '4q', '5fs', '8)', '8-d', '8-o', '86', '8d', ':###..', ':$', ':&', ":'(", ":')", ":'-(", ":'-)", ':(', ':)', ':*', ':-###..', ':-&', ':-(', ':-)', ':-))', ':-*', ':-,', ':-.', ':-/', ':-<', ':-d', ':-D', ':-o', ':-p', ':-[', ':-\\', ':-c', ':-p', ':-|', ':-||', ':-Þ', ':/', ':3', ':<', ':>', ':?)', ':?c', ':@', ':d', ':D', ':l', ':o', ':p', ':s', ':[', ':\\', ':]', ':^)', ':^*', ':^/', ':^\\', ':^|', ':c', ':c)', ':o)', ':o/', ':o\\', ':o|', ':P', ':{', ':|', ':}', ':Þ', ';)', ';-)', ';-*', ';-]', ';d', ';D', ';]', ';^)', '</3', '<3', '<:', '<:-|', '=)', '=-3', '=-d', '=-D', '=/', '=3', '=d', '=D', '=l', '=\\', '=]', '=p', '=|', '>-:', '>.<', '>:', '>:(', '>:)', '>:-(', '>:-)', '>:/', '>:o', '>:p', '>:[', '>:\\', '>;(', '>;)', '>_>^', '@:', '@>-->--', "@}-;-'---", 'aas', 'aayf', 'afu', 'alol', 'ambw', 'aml', 'atab', 'awol', 'ayc', 'ayor', 'aug-00', 'bfd', 'bfe', 'bff', 'bffn', 'bl', 'bsod', 'btd', 'btdt', 'bz', 'b^d', 'cwot', "d-':", 'd8', 'd:', 'd:<', 'd;', 'd=', 'doa', 'dx', 'ez', 'fav', 'fcol', 'ff', 'ffs', 'fkm', 'foaf', 'ftw', 'fu', 'fubar', 'fwb', 'fyi', 'fysa', 'g1', 'gg', 'gga', 'gigo', 'gj', 'gl', 'gla', 'gn', 'gr8', 'grrr', 'gt', 'h&k', 'hagd', 'hagn', 'hago', 'hak', 'hand', 'heart', 'hearts', 'hho1/2k', 'hhoj', 'hhok', 'hugz', 'hi5', 'idk', 'ijs', 'ilu', 'iluaaf', 'ily', 'ily2', 'iou', 'iyq', 'j/j', 'j/k', 'j/p', 'j/t', 'j/w', 'j4f', 'j4g', 'jho', 'jhomf', 'jj', 'jk', 'jp', 'jt', 'jw', 'jealz', 'k4y', 'kfy', 'kia', 'kk', 'kmuf', 'l', 'l&r', 'laoj', 'lmao', 'lmbao', 'lmfao', 'lmso', 'lol', 'lolz', 'lts', 'ly', 'ly4e', 'lya', 'lyb', 'lyl', 'lylab', 'lylas', 'lylb', 'm8', 'mia', 'mml', 'mofo', 'muah', 'mubar', 'musm', 'mwah', 'n1', 'nbd', 'nbif', 'nfc', 'nfw', 'nh', 'nimby', 'nimjd', 'nimq', 'nimy', 'nitl', 'nme', 'noyb', 'np', 'ntmu', 'o-8', 'o-:', 'o-|', 'o.o', 'O.o', 'o.O', 'o:', 'o:)', 'o:-)', 'o:-3', 'o:3', 'o:<', 'o;^)', 'ok', 'o_o', 'O_o', 'o_O', 'pita', 'pls', 'plz', 'pmbi', 'pmfji', 'pmji', 'po', 'ptl', 'pu', 'qq', 'qt', 'r&r', 'rofl', 'roflmao', 'rotfl', 'rotflmao', 'rotflmfao', 'rotflol', 'rotgl', 'rotglmao', 's:', 'sapfu', 'sete', 'sfete', 'sgtm', 'slap', 'slaw', 'smh', 'snafu', 'sob', 'swak', 'tgif', 'thks', 'thx', 'tia', 'tmi', 'tnx', 'true', 'tx', 'txs', 'ty', 'tyvm', 'urw', 'vbg', 'vbs', 'vip', 'vwd', 'vwp', 'wag', 'wd', 'wilco', 'wp', 'wtf', 'wtg', 'wth', 'x-d', 'x-p', 'xd', 'xlnt', 'xoxo', 'xoxozzz', 'xp', 'xqzt', 'xtc', 'yolo', 'yoyo', 'yvw', 'yw', 'ywia', 'zzz', '[-;', '[:', '[;', '[=', '\\-:', '\\:', '\\:<', '\\=', '\\^:', '\\o/', '\\o:', ']-:', ']:', ']:<', '^<_<', '^urs', '{:', '|-0', '|-:', '|-:>', '|-o', '|:', '|;-)', '|=', '|^:', '|o:', '||-:', '}:', '}:(', '}:)', '}:-(', '}:-)', 'x-d', 'x-p', 'xd', 'xp', 'yay']
        words = text.split()
        neat_words = []
        for word in words:
            if word not in words_to_remove:
                neat_words.append(word)
        return ' '.join(neat_words)

    def remove_emojis(self, text):
        # Define regex pattern to match emojis
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        # Remove emojis from the text
        cleaned_text = emoji_pattern.sub(r'', text)
        return cleaned_text

    def is_special_characters(self, sentence):
        if len(sentence) < 5:
            # Get the set of all punctuation characters
            punctuation_chars = set(string.punctuation)

            # Check if all characters in the string are punctuation, special characters, or empty spaces
            for char in sentence:
                if char not in punctuation_chars and not char.isspace():
                    return False
            return True
        return False

    def remove_nan(self, text):
        return text

    def list_to_file(self, str_list, language='english', tokenize_sentences = False):
        if(tokenize_sentences):
            long_str = ''
            for item in str_list:
                sentences = DataPreprocessing.tokenize_review_into_sentences_nltk(item, language)
                review_str = '\n'.join(sentences)
                review_str = review_str + ' DOC_SEPARATOR \n'
                long_str = long_str + review_str
        else:
            long_str = '\n'.join(str_list)
        file = io.StringIO(long_str)
        return file.read()

    @staticmethod
    def tokenize_review_into_sentences_nltk(review, language='english'):
        """
        Tokenizes a review into sentences for the specified language.

        Parameters:
        review (str): The review text to be tokenized.
        language (str): The language of the review text ('english' or 'spanish').

        Returns:
        list: A list of sentences.
        """
        sentences = sent_tokenize(review, language=language)
        # Filter out items that contains puntuations only
        #filtered_sentences = [sentence for sentence in sentences if sentence not in ['.', '?', '!', '. )']]
        filtered_sentences = [sentence for sentence in sentences if len(sentence) > 3]
        return filtered_sentences

    def get_data(self, data_file = ""):
        file_extension = self.get_file_extension(data_file)

        if(file_extension in ['.xls', '.xlsx']):
            dfTrain = pd.read_excel(data_file)
        #elif(file_extension in ['.csv']):
        #    dfTrain = pd.read_csv(data_file)
        else:
            print("Supported file formats are: xls, xlsx")
            sys.exit(1)  # Exit the script if the file path is empty

        dfTrain['reviewCorr'] = dfTrain['Review'].apply(self.remove_words_and_emojis)
        dfTrain['reviewCorr'] = dfTrain['reviewCorr'].apply(self.remove_emojis)
        dfTrain['reviewCorr'] = dfTrain['reviewCorr'].apply(self.fix_punctuation)

        # *********** START - batchSpacy Index Mismach Issue Patch  ***********
        # Apply the function to the 'Text' column and create a boolean mask
        mask = dfTrain['reviewCorr'].apply(self.is_special_characters)
        # Filter out rows where the mask is True
        dfTrain = dfTrain[~mask]
        # Reset the index of the filtered DataFrame
        dfTrain.reset_index(drop=True, inplace=True)
        # *********** END - batchSpacy Index Mismach Issue Patch  ***********

        return dfTrain

    def batchSpacySplit(self, text_data, lang='en'):
        response = []
        try:
            spacy.prefer_gpu()
            if(lang == 'en' or lang == 'english'): 
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            elif(lang == 'es' or lang == 'spanish'): 
                nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            #nlp.add_pipe("sentencizer")
            config = {"punct_chars": ['\n', 'DOC_SEPARATOR']}
            nlp.add_pipe("sentencizer", config=config)
            #nlp.add_pipe("custom_sentencizer", before="parser")  # Insert before the parser
            nlp.max_length = 1500000000

            #for data_item in text_data:
            response = []
            sentences = []
            doc = nlp(text_data)
            for sent in doc.sents:
                #print("sent.text ", sent.text)
                if sent.text == '\n' or sent.text == '' or sent.text == ' ' or sent.text == '.' or sent.text == '!':
                    pass
                else:
                    tokens = []
                    postags = []
                    lemmas = {}
                    for token in sent:
                        if(token.text == "DOC_SEPARATOR"):
                            res = {"tokens": tokens, "postags": postags, "lemmas": lemmas}
                            sentences.append(res)
                            response.append(sentences)
                            tokens = []
                            postags = []
                            lemmas = {}
                            sentences = []
                            break # End of Review / Document
                        lemmas.setdefault(token.text, token.lemma_) 
                        tokens.append(token.text)
                        postags.append(token.pos_)
                    res = {"tokens": tokens, "postags": postags, "lemmas": lemmas}

                    if(len(tokens) > 0):
                        sentences.append(res)
                #response.append(sentences)
        except Exception as error:
            print(error)
        return response
    
    def batchSpacy(self, text_data, lang='en'):
        response = []
        try:
            spacy.prefer_gpu()
            if(lang == 'en'): 
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            elif(lang == 'es'): 
                nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            config = {"punct_chars": ['\n']}
            nlp.add_pipe("sentencizer", config=config)
            #nlp.add_pipe("custom_sentencizer", before="parser")  # Insert before the parser
            nlp.max_length = 1500000000

            doc = nlp(text_data)
            for sent in doc.sents:
                if sent.text == '\n':
                    pass
                else:
                    tokens = []
                    postags = []
                    lemmas = {}
                    for token in sent:
                        lemmas.setdefault(token.text, token.lemma_) 
                        tokens.append(token.text)
                        postags.append(token.pos_)
                    res = {"tokens": tokens, "postags": postags, "lemmas": lemmas}
                    response.append(res)
        except Exception as error:
            print(error)
        return response

    def get_batch_spacy(self, clearn_str_list, lang='en', split_review=True):
        if(lang == 'es'):
            language = 'spanish'
        else:
            language = 'english'
        
        batch_size = 10000
        # Generator comprehension
        batches = (clearn_str_list[i:i + batch_size] for i in range(0, len(clearn_str_list), batch_size))
        batch_spacy = []
        count = 0
        for batch in batches:
            count = count + 1
            batch_file_data = self.list_to_file(batch, language, split_review)
            if(split_review):
                spacy_single_batch = self.batchSpacySplit(batch_file_data, lang)
            else:
                spacy_single_batch = self.batchSpacy(batch_file_data, lang)
            batch_spacy.extend(spacy_single_batch)
        return batch_spacy
    
class EnglishSentimentAnalysis:

    _instance = None
   
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(cls, "Object Created!")

        return cls._instance
    
    # List of Negataions
    lstNeg = \
        ['nor', 'seldom', 'arent', "won't", 'rarely', "doesn't", 'barely', 'mustnt', "hadn't",
        "don't", 'isnt', 'nowhere', 'cannot', 'doesnt', "couldn't", 'oughtnt', 'couldnt', "haven't",
        'never', 'no_one', "ain't", 'despite', "daren't", 'without', 'uhuh', 'scarcely', 'nope',
        'wouldnt', 'neither', "mustn't", 'nothing', "needn't", 'dont', "isn't", "shouldn't", 'not',
        'uh-uh', 'wasnt', 'neednt', 'werent', "oughtn't", "wouldn't", 'cant', 'nobody', "hasn't",
        "weren't", "shan't", 'darent', 'shouldnt', 'aint', 'shant', 'hadnt', 'didnt', "mightn't",
        'none', 'wont', "aren't", "can't", "didn't", 'mightnt', 'havent', "wasn't", 'no', 'hardly', 'hasnt'
    ]

    mode = 'Model'
    dAdj = {}
    dAdv = {}
    dNoun = {}
    dVerb = {}
    dInt = {}

    device = ''
    tokenizer = ''
    model = ''
    dec_tree = []
    config = ''
    # nlp = None
    # nltk = None

    sentdicts = {"ADV": dAdv, "ADJ": dAdj, "NOUN": dNoun, "VERB": dVerb}

    def __init__(self):
        # Load applicataion config file
        self.config = dotenv_values("config.env")

        # Stanza Nlp
        stanza.download('en')
        self.nlp = stanza.Pipeline('en')

        # # Download the necessary NLTK resources
        # self.nltk = nltk.download('punkt')

        if(self.config['DEFAULT_DIC'] == 'DIC1'): # SO-CAL Dictionary
            self.dAdj = self.populateDictComp(self.config['ENGLISH_DIC1_DADJ'])
            self.dAdv = self.populateDictComp(self.config['ENGLISH_DIC1_DADV'])
            self.dNoun = self.populateDictComp(self.config['ENGLISH_DIC1_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['ENGLISH_DIC1_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC2'): # Merged (SO-CAL + VADER)
            self.dAdj = self.populateDictComp(self.config['ENGLISH_DIC2_DADJ'])
            self.dAdv = self.populateDictComp(self.config['ENGLISH_DIC2_DADV'])
            self.dNoun = self.populateDictComp(self.config['ENGLISH_DIC2_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['ENGLISH_DIC2_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC3'): # Rest-Mex Only
            self.dAdj = self.populateDictComp(self.config['ENGLISH_DIC3_DADJ'])
            self.dAdv = self.populateDictComp(self.config['ENGLISH_DIC3_DADV'])
            self.dNoun = self.populateDictComp(self.config['ENGLISH_DIC3_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['ENGLISH_DIC3_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC4'): # Merged (Rest-Mex + SO-CAL + VADER)
            self.dAdj = self.populateDictComp(self.config['ENGLISH_DIC4_DADJ'])
            self.dAdv = self.populateDictComp(self.config['ENGLISH_DIC4_DADV'])
            self.dNoun = self.populateDictComp(self.config['ENGLISH_DIC4_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['ENGLISH_DIC4_DVERB'])

        # Intensifiers Dictionary
        self.dInt = self.populateDictComp(self.config['ENGLISH_INTENSIFIERS'])

        # Manually Fixe Errors in Sentiment Dictionaries
        # dNoun['service'] = 2
        # dAdj['worst'] = -5
        # dAdj['best'] = 5
        # dAdj['basic'] = -1
        # #dVerb['gustar'] = 5
        # #dNoun['mínimo'] = -5
        # #dNoun['coña']: -5
        # #dNoun['montón']: 5
        # dAdj['moderate']: 1

        # Extra Patch - Fix Issue Rules Overwritten
        dictionaries = [self.dAdj, self.dNoun, self.dVerb, self.dAdv]
        # Iterate through each dictionary
        for d in dictionaries:
            # Iterate through the keys of dInt
            for key in self.dInt.keys():
                # Check if the key exists in the current dictionary
                if key in d:
                    # Delete the entry from the current dictionary
                    del d[key]

        self.sentdicts = {"ADV": self.dAdv, "ADJ": self.dAdj, "NOUN": self.dNoun, "VERB": self.dVerb}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForTokenClassification.from_pretrained(self.config['ENGLISH_PARSER_PATH'])

    #Process the dictionaries
    def populateDictComp(self, fileLocation):
        dic = {}
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(fileLocation, 'r', encoding=encoding) as file:
                    for line in file:
                        try:
                            #(key,value) = line.split()
                            #dct[key]=float(value.strip())
                            word, value = line.strip().split('\t')
                            dic[word] = float(value)
                        except ValueError:
                            pass
                            #print("Error parsing line:", line)
            except UnicodeDecodeError:
                #pass
                print(f"Failed to decode using {encoding} encoding")
                print("Failed to decode the file using any of the specified encodings")
        return dic

    def populateDict(self, fileLocation):
        dic = {}
        f = open(fileLocation, encoding='utf-8')
        for line in f:
            words = line.split()
            if(len(words)>2):
                keys = words[:-1]
                key = '-'.join(keys)
                value = words[-1]
                dic[key]=float(value.strip())
            else:
                (key,value) = line.split()
                dic[key] = float(value.strip())
        return dic

    # analysing a string (=sent) with Stanza
    def createDicStanza(self, sent):
        if(isinstance(sent, list)):
            sent = sent[0]
        doc     = self.nlp(sent)
        dicts   = doc.to_dict()
        return dicts

    def createDicModel(self, text, spacy_datas, lang='en'):
        try:
            spacy_lemmas = {}
            dec_tree = ''

            for spacy_data in spacy_datas:
                tokens_with_spacy = spacy_data['tokens']
                if tokens_with_spacy[-1] == '\n':
                    tokens_with_spacy = tokens_with_spacy[:-1]
                
                tokenized_text = pre_tokenized_text(tokens_with_spacy)
                #print("tokenized_text ", tokenized_text)
                # Example pre-tokenized text/data
                #tokenized_text = ["[CLS]", 'I', 'do', "n't", 'eat', 'very', 'much', 'pieces', 'of', 'chocolate', '.', "[SEP]"]

                # Convert tokens to input IDs using the tokenizer
                #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                #print("input_ids ", input_ids)

                # Create input tensors
                input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add a batch dimension
                #print("input_ids ", input_ids)
                
                with torch.no_grad():
                    #logits = model(**inputs).logits
                    logits = self.model(input_ids).logits

                    predictions = torch.argmax(logits, dim=2)
                    # Model's predictions/rules
                    predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0][1:-1]]

                words = tokens_with_spacy
                postags = spacy_data['postags']
                #print("postags ", postags)
                if postags[-1] == 'SPACE':
                    postags = postags[:-1]
                #print(postags)
                labels = []
                for p in predicted_token_class:
                    if(p != '[CLS]' and p != '[SEP]'):
                        labels.append(D_Label.from_string(p, sep="_"))
                #print("labels ", labels)
                lin_tree = LinearizedTree(words=words, postags=postags,
                            additional_feats=[], labels=labels, n_feats=0)
                #print("lin_tree ", lin_tree)
                #encoder = D_PosBasedEncoding(separator="_")
                encoder = D_NaiveRelativeEncoding(separator="_", hang_from_root=True)
                #encoder = D_NaiveAbsoluteEncoding(separator="_")
                dec_tree = encoder.decode(lin_tree)
                dec_tree.postprocess_tree(search_root_strat=D_ROOT_HEAD, allow_multi_roots=False)
                dec_tree = str(dec_tree)
                #print("dec_tree", dec_tree)
                spacy_lemmas = spacy_data['lemmas']
                #print("spacy_lemmas ", spacy_lemmas)
            return tree2dic_optim(dec_tree, spacy_lemmas), dec_tree
        except Exception as e:
            print("Model Inference Error: " + str(e))

    def FilterNonLemmaWords(self, Sentence): # this function deletes problematic lemmas like Spanish: hacerlo
        FilteredSentence = [word for word in Sentence if 'lemma' in word.keys()]
        return (FilteredSentence)
    
    def CreateDefaultElementType(self, Sentence): # assigns to each lemma a default element type 'ord'
        for i in range(len(Sentence)):
            Sentence[i]['elementType'] = 'ord'
        return (Sentence)

    # Set element score from Sentiment dictionaries
    def GetElementScore(self, Sentence):
        for i in range(len(Sentence)):
            lem = Sentence[i]['lemma']
            upos = Sentence[i]['upos']

            if upos == 'ADJ' and lem in self.dAdj.keys():
                Sentence[i]['elementScore'] = self.dAdj[lem]
            elif upos == 'ADV' and lem in self.dAdv.keys():
                Sentence[i]['elementScore'] = self.dAdv[lem]
            elif upos == 'VERB' and lem in self.dVerb.keys():
                Sentence[i]['elementScore'] = self.dVerb[lem]
            elif upos == 'NOUN' and lem in self.dNoun.keys():
                Sentence[i]['elementScore'] = self.dNoun[lem]
            else:
                Sentence[i]['elementScore'] = 'none'
        return Sentence

    def InitializeSentimentScore(self, Sentence): # default is a Sentiment Score
        for i in range(len(Sentence)):
            Sentence[i]['SentimentScore'] = 'none'
        return Sentence

    def GetSentenceWords(self, Sentence): # important for the demo
        #Sentence is a list of word dictionaries
        lstwords = []
        for word in Sentence:
            lstwords.append(word["text"])
        return (lstwords)

    def getChildParentDicts(self, dct):
        dctChild       = {} ## for now, this dictionary is important
        dctParent      = {}
        dctBeforeToken = {}

        for dicVal in dct:
            #print(dicVal)
            elementID     = dicVal['id']
            elementHead   = dicVal['head']
            elementLemma  = dicVal['lemma']

            #print(type(dct))
            if(elementID != 1 ):
                eleBeforeDic = dct[int(elementID)-2]
                elementBeforLemma = eleBeforeDic['lemma']
                #print(elementID, elementLemma, eleBeforeDic, eleBeforeDic['lemma'], elementBeforLemma)
            else:
                elementBeforLemma = "-"

            # dic parent is done
            dctParent[elementID] = elementHead
            dctBeforeToken[elementLemma] = elementBeforLemma

            if elementHead not in dctChild.keys():
                dctChild[elementHead] = []

            # Adding to the parent node (Elements and ElementsLemma)
            dctChild[elementHead].append(elementID)

        # working on Sibling dictionary
        dctSibling = {}

        # Giviing some default values
        for key in dctParent.keys():
            dctSibling[key] = []
        for key in dctChild.keys():
            childList = dctChild[key]
            #print(childList)
            for chld in childList:
                #print("Child is: ", chld)
                dctSibling[chld] = [x for x in childList if x != chld]
        return dctChild, dctParent, dctSibling, dctBeforeToken

    def calcbranchorder(self, dctChild):
        if list(dctChild.keys()) == [0]:
            calcheads=[0]
        else:
            allheads = list(dctChild.keys())
            calcheads = []
            #children = dctChild[0]
            children = dctChild.get(0)  # Returns None if key doesn't exist
            children = children or []   # Assuming you want an empty list if children is None
            nextchildren = []
            numchildren = len(children)
            #print("numchildren is", numchildren)
            while numchildren > 0:
                nextchildren = []
                #print('numchildren = ', numchildren)
                for child in children:
                    #print('child = ', child)
                    if child in allheads:
                        #print('headchild = ', child)
                        calcheads.append(child)
                        nextchildren = nextchildren + dctChild[child]
                        #print('calcheads = ', calcheads)
                        #print('nextchildren = ', nextchildren)
                children = nextchildren
                numchildren = len(children)
            calcheads.reverse()
        #print('branch order = ', calcheads)
        return calcheads

    def calcbranch(self, dct, headId, childIds):
        dInt        = self.dInt
        lstNeg      = self.lstNeg
        sentdicts   = self.sentdicts

        #dct is a list of word dictionaries like Sentence
        lstchildren = [dct[childId-1]["text"] + "({},{})".format(childId, dct[childId-1]["deprel"]) for childId in childIds]
        #lstchildren=[dct[childId-1]["deprel"] + "({})".format(childId) for childId in childIds]
        #print("head: {}({}), children: {}".format(dct[headId-1]["text"], headId, lstchildren) )
        
        ####finding relevant elements in each branch
        a           = 0
        b           = 0
        c           = 0 # new 3.6.
        neg         = 0
        p           = 0
        pvalue      = 0.25
        negvalue    = -4
        Negstat     = False
        fmodstat    = False
        lexmodstat  = False
        conjmodstat=False # new 3.6.
        lexstat     = False
        lstcntchildren  = []
        lstmodification = []
        lstnegation     = []
        PrintList       = []
        #####create default outputs
        HeadOutput      = []
        NegOutput       = []
        ModOutput       = []

        #### starting with children of the branch
        PrintList.append('Number of Children: {}'.format(len(childIds)))
        for i in range(len(childIds)):
        #for childId in childIds:###
            #print("childid=", childId)
            childId = childIds[i]
            lemma = dct[childId-1]["lemma"]
            text = dct[childId-1]["text"]
            pos = dct[childId-1]["upos"]#
            deprel = dct[childId-1]["deprel"] # head, amod, nmod, mod in deprel
            elementScore = dct[childId-1]["elementScore"]
            if dct[childId-1]["SentimentScore"] != 'none':
                SentimentScore = dct[childId-1]["SentimentScore"]
            else:
                SentimentScore = elementScore
            #lstchildren[i]=lstchildren[i] + "({})".format(deprel)

            if lemma in lstNeg or text in lstNeg: # find negation
                ### Child cannot be a head. ###
                Negstat = True
                neg = negvalue
                lstchildren[i] = lstchildren[i] + "({})".format("N")
                dct[childId-1]['elementType'] = 'neg'
                #print("negation token is: ", text, "negation childid is: ", childId, "the negation value is: ", neg) #information for the demo
                PrintList.append("Child{}: '{}'({}), Neg={}".format(i, text, childId, neg)) # important for the demo
                NegOutput = [text,neg]
            #elif lemma in dInt.keys() and 'mod' in deprel:# new added 26 May,  find intensification
            elif lemma in dInt.keys() and ('mod' in deprel or 'det' in deprel):  # update on 3.6.
                ### Child cannot be a head. ###
                fmodstat = True
                PrevExists = (childId-2) >= 0
                NextExists = childId <= len(dct)-1
                if PrevExists:
                    PrevIsInt = dct[childId - 2]["lemma"] in dInt.keys()
                    PrevIsNeg = dct[childId - 2]["lemma"] in lstNeg
                    PrevIsHead = dct[childId - 2]["head"] == dct[childId-1]["id"]
                else:
                    PrevIsInt = False
                    PrevIsNeg = False
                    PrevIsHead = False
                if NextExists:
                    NextIsInt = dct[childId]["lemma"] in dInt.keys()
                else:
                    NextIsInt = False
                if PrevIsInt == False and NextIsInt == False: # single intensifier found, e.g. a VERY good service
                    if PrevIsNeg and PrevIsHead: # looking for head negation of a single intensifier, e.g. not many problems (Stanza analyses not as a head of many)
                        if dInt[lemma] > 0:
                            b = -abs(dInt[lemma])
                        if dInt[lemma] < 0:
                            b = abs(dInt[lemma])
                    else:
                        b = dInt[lemma]
                    #print("{}single modifier={}({}), b={}".format(indent, text, childId, b)) ###new 23.5
                    PrintList.append("single modifier={}({}), b={}".format(text, childId, b)) ###new 23.5
                    PrintList.append("Child{}: '{}'({}), 1b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(1b={})".format(b)
                    ModOutput = [text,b]
                elif PrevIsInt: #double modifier found, e.g. little bit, very small, etc.
                    lemma2 = dct[childId - 2]["lemma"]
                    fmodscore = dInt[lemma]*(1+dInt[lemma2]) # a compound b or 2b
                    b = fmodscore
                    #print("double modifier is:", text, "childid is: ", childId, "the value of double modifier is: ",  b) ###new 22.5
                    PrintList.append("double modifier={}({}), b={}".format(text, childId, b)) ###new 23.5, double modifier
                    PrintList.append("Child{}: '{}'({}), 2b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(2b={})".format(b)
                    ModOutput = ["{}, {}".format(dct[childId - 2]["text"], text), b]
                else:
                    PrintList.append("Child{}: '{}'({}), first part of 2b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(2b={})".format(b)


            elif (lemma in sentdicts.keys() or SentimentScore!='none') and 'cl' not in deprel and 'conj' not in deprel: #new update on 3.6= children are neither adj, verbs, nouns or adverbs, but pronouns with a SentimenSscore
                lexmodstat = True
                a = SentimentScore
                lstchildren[i] = lstchildren[i] + "(a={})".format(SentimentScore)
                    #PrintList.append("Child{}: '{}'({}), elementScore={}, SentimentScore, a={}".format(i, text, childId, elementScore, SentimentScore))
            elif (lemma in sentdicts.keys() or SentimentScore!='none') and ('cl' in deprel or 'conj' in deprel):  # new update on 3.6.
                conjmodstat = True
                c = SentimentScore
                lstchildren[i] = lstchildren[i] + "(c={})".format(SentimentScore)
            elif lemma in ['but', 'although', 'however', 'nevertheless', 'nonetheless']:
                ### Child cannot be a head. ###
                p = pvalue
                lstchildren[i] = lstchildren[i] + "(p={})".format(p)
                #print("but is present and has the value of: ", p) # for the demo
                #PrintList.append("{} is present, p=".format(lemma, p)) # for the demo
                PrintList.append("Child{}: '{}'({}), p={}".format(i, text, childId, p))
            else:
                lstchildren[i] = lstchildren[i] + "({}={})".format(pos,SentimentScore)
                PrintList.append("Child{}: '{}'({})".format(i, text, childId))
            #print("child: ", childId, text, elementScore)
        PrintList.append("Children-derived score elements: a={}, b={}, neg={}, p={}".format(a, b, neg, p)) # relevant for demo
    
        ######################################################################################################################################
        ####continuing with parents/heads of the branch ###
        lemma = dct[headId-1]["lemma"]
        pos = dct[headId-1]["upos"]
        deprel = dct[headId-1]["deprel"]
        text = dct[headId-1]["text"]
        #headmodstat = "mod" in deprel or 'conj' in deprel
        headmodstat = "mod" in deprel
        #headmodstat = "mod" in deprel or 'DET' in 'upos' #new
        headscore = "none"
        headsentimentscore = 'none'
        childp = p
        head_a = 0
        head_b = 0
        head_neg = 0
        head_p = 0
        calc_a, calc_b, calc_neg, calc_p = 0, 0, 0, 0
        words_to_match = ["bit", "lot", "bunch"]
        lstheads=[]

        if lemma == "nothing":
            head_neg = negvalue
            headsentimentscore = (a * (1+b)  + (np.sign(a*(1+b)))* head_neg)*(1+p) # new update on 3rd of June
            dct[headId-1]["SentimentScore"] = headsentimentscore #new
            PrintList.append("Head: '{}'({}), 'nothing'={}, SentimentScore={}".format(text, headId, head_neg, headsentimentscore))
            lstheads.append([text,head_neg, headsentimentscore])
            #calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
        # Example condition
        elif lemma in words_to_match and pos =="NOUN":
        #elif lemma in dInt.keys(): #new 14.5. accounting for intensifiers as heads as in "a bunch of problems" or 'a bit of heaven'
            fmodscore = dInt[lemma]
            head_b = fmodscore
            fmodstat = True
            if True: #all([(type(x) is int) or (type(x) is float) for x in [a, head_b, p, neg]]):
                headsentimentscore = (a * (1+head_b)  + (np.sign(a*(1+head_b)))* neg)*(1+p) # new 25.5
                #headsentimentscore=(childsentimentscore * (1+head_b)  + (np.sign(childsentimentscore*(1+head_b)))* neg)*(1+p)# new 26.5
                dct[headId-1]["SentimentScore"] = headsentimentscore #new
                PrintList.append("Head: '{}'({}), head intensifier, b={}, SentimentScore={}".format(text, headId, head_b, headsentimentscore))
                HeadOutput = [text, a, headsentimentscore]
                #print("HeadOutput", HeadOutput)
                #calc_a, calc_b, calc_neg, calc_p = a, head_b, neg, p
            else:
                PrintList.append("Head: '{}'({}), a, b, p or neg is not a number, SentimentScore={}".format(text, headId, headsentimentscore))
                calc_a, calc_b, calc_neg, calc_p = a, head_b, neg, p

        elif lexmodstat and conjmodstat: # if one of the children is a lexical head, then do the calculation of headsentimentscore.
            if True: #all([(type(x) is int) or (type(x) is float) for x in [a, b, p, neg]]):
                calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                headsentimentscore = np.mean([((a * (1+b)  + (np.sign(a*(1+b)))* neg)*(1+p)), c]) # new 3.6.
                #headsentimentscore=(childsentimentscore * (1+b)  + (np.sign(childsentimentscore*(1+b)))* neg)*(1+p) # 26.5
                dct[headId-1]["SentimentScore"] = headsentimentscore # the head gets the polarity score of the child
                PrintList.append("Head: '{}'({}), child is a sentiment word with a conjunction, a={},c={}, SentimentScore={}".format(text, headId, a, c, headsentimentscore)) # 25.5.
                #PrintList.append("Head inheriting Childfeatures: '{}'({}), childsentimentscore={}, SentimentScore={}".format(text, headId, childsentimentscore, headsentimentscore)) # 26.5.
                #calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                HeadOutput = [text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
            else:
                PrintList.append("Head: '{}'({}), a, b, p or neg is not a number, SentimentScore={}".format(text, headId, a, headsentimentscore))
        elif lexmodstat and conjmodstat != True: # if one of the children is a lexical head, then do the calculation of headsentimentscore.
            if True: #all([(type(x) is int) or (type(x) is float) for x in [a, b, p, neg]]):
                calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                headsentimentscore = (a * (1+b)  + (np.sign(a*(1+b)))* neg)*(1+p)#25.5.
                #headsentimentscore=(childsentimentscore * (1+b)  + (np.sign(childsentimentscore*(1+b)))* neg)*(1+p) # 26.5
                dct[headId-1]["SentimentScore"] = headsentimentscore # the head gets the polarity score of the child
                PrintList.append("Head: '{}'({}), child is a sentiment word, a={}, SentimentScore={}".format(text, headId, a, headsentimentscore)) # 25.5.
                #PrintList.append("Head inheriting Childfeatures: '{}'({}), childsentimentscore={}, SentimentScore={}".format(text, headId, childsentimentscore, headsentimentscore)) # 26.5.
                #calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                HeadOutput = [text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
            else:
                PrintList.append("Head: '{}'({}), a, b, p or neg is not a number, SentimentScore={}".format(text, headId, a, headsentimentscore))
        elif pos in sentdicts.keys() and conjmodstat != True: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
            dsent = sentdicts[pos]
            if lemma in dsent.keys():
                headscore = dsent[lemma]
                head_a = dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                headsentimentscore = (head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)
                PrintList.append("Head: '{}'({}), head is a sentiment word, head_a={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput = [text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"] = headsentimentscore
                #childp=0
        elif pos in sentdicts.keys() and conjmodstat: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
            dsent = sentdicts[pos]
            if lemma in dsent.keys(): #or SentimentScore!='none':
                headscore = dsent[lemma]
                head_a = dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                headsentimentscore = np.mean([((head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)), c])
                PrintList.append("Head: '{}'({}), head is a sentiment word with a conjunction head, head_a={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput = [text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"] = headsentimentscore
                #childp=0
            else: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
                headsentimentscore = c
                #headscore=dsent[lemma]
                #head_a=dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                #headsentimentscore=np.mean([((head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)), c])
                PrintList.append("Head: '{}'({}), head has no sentiment word with a conjunction head, c={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput = [text, c, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"]= headsentimentscore
        else:
            PrintList.append("Head: '{}'({}), No calculation possible, SentimentScore={}".format(text, headId, headsentimentscore))
        #PrintList.append("Head-derived score elements: head_a={}, head_b={}, head_neg={}, head_p={}".format(head_a, head_b, head_neg, head_p))
        #PrintList.append("Calculation score elements: calc_a={}, calc_b={}, calc_neg={}, calc_p={}".format(calc_a, calc_b, calc_neg, calc_p))
        #PrintList.append('Result of Calculation:')
        #PrintList.append("head: {}({}), children: {}".format(dct[headId-1]["text"], headId, lstchildren))
        #print("list of heads:", lstheads)
        #print("list of negation:", lstnegation)
        #print("list of modification:", lstmodification)
        #print("ListChildren is:", lstchildren) #new 27.5.
        #print("HeadOutput", HeadOutput)
        #indent = "      " #indent each statement by this much.
        #for x in PrintList:
            #print(indent + x)
        return dct, HeadOutput, NegOutput, ModOutput
    
    def calcSentenceScore(self, dct): # dct= Sentence which is a list of word dictionaries and the function takes only one sentence
        lstScores = []
        lstHeadOutput = []
        lstModOutput = []
        lstNegOutput = []

        try:
            dctChild, dctParent, dctSibling, dctBeforeToken = self.getChildParentDicts(dct)
            #print(dctChild)

            ###Step 2 figure out order of nodes
            branchheadIds = self.calcbranchorder(dctChild)
            #print("branchheadIds: {}".format(branchheadIds))
            #topheadid = branchheadIds[-1] # new updated on 22nd of May.
            topheadid = None if not branchheadIds else branchheadIds[-1]
            #topheadid = branchheadIds[-1]

            ## Step 3 looping over nodes
            #print('Number of branches: {}'.format(len(branchheadIds)))
            if branchheadIds != [0] and len(branchheadIds) > 0: # calculate branches that have more than one word.
            #if branchheadIds != [0]: # calculate branches that have more than one word.
                for BranchIndex in range(len(branchheadIds)): #looping over nodes
                    headId = branchheadIds[BranchIndex]
                    print('Branch{}'.format(BranchIndex))
                    print("Branch{} = head: '{}', children: {}".format(BranchIndex, dct[headId-1]['text'],[dct[Id-1]["text"] for Id in dctChild[headId]]))
                    #dct=calcbranch(dct, headId, dctChild[headId])
                    dct, HeadOutput, NegOutput, ModOutput = self.calcbranch(dct, headId, dctChild[headId])
                    #if
                    lstHeadOutput.append(HeadOutput)
                    lstModOutput.append(ModOutput)
                    lstNegOutput.append(NegOutput)
                    
            ###Step 4 collect the scores of branchheadIds
                    #if dct[headId-1]["elementScore"]!="none":
                    #if dct[headId-1]["SentimentScore"]!="none":
                    if headId == topheadid and dct[headId-1]["elementScore"] != "none":
                        #lstScores.append(dct[headId-1]["elementScore"])
                        lstScores.append(dct[headId-1]["SentimentScore"])
                        #print("lstScores", lstScores)
            else: # calculate branches with only one word
                headId = 1 #headId=0, ChildId=[1] dctChildren= {0: [1]}
                #dct=calcbranch(dct, headId, [])
                dct, HeadOutput, NegOutput, ModOutput = self.calcbranch(dct, headId, [])
                lstHeadOutput.append(HeadOutput)
                lstModOutput.append(ModOutput)
                lstNegOutput.append(NegOutput)
                #dct=calcbranch(dct, headId, dctChild[headId]) #new
                print("elementScore:{}".format(dct[headId - 1]["elementScore"]))
                print("SentimentScore:{}".format(dct[headId - 1]["SentimentScore"]))
            ###Step 4 collect the scores of branchheadIds
            if dct[headId - 1]["SentimentScore"] != "none": # dct[headId - 1]["elementScore"] does not exist because it is either none nor some score
                print("elementScore:{}".format(dct[headId - 1]["elementScore"]))
                print("SentimentScore:{}".format(dct[headId - 1]["SentimentScore"]))
                #if dct[headId - 1]["elementScore"]=="none":
                #lstScores.append(dct[headId - 1]["elementScore"])
                lstScores.append(dct[headId - 1]["SentimentScore"])
                #print("lstScores", lstScores)
            else:
                lstScores.append('none')
        except Exception as e:
            print("Sentence Score Calculation Error: " + str(e))
            lstScores.append('none')
        ###Step 5 create a sentence score
        print("lstScores: {}".format(lstScores))

        if len(lstScores) != 0 and any([score != 'none' for score in lstScores]):
            SentenceScore = np.mean([float(score) for score in lstScores if score != 'none'])
            return SentenceScore, lstHeadOutput, lstModOutput, lstNegOutput
        else:
            return 'none', lstHeadOutput, lstModOutput, lstNegOutput

    def calcReviewScore(self, review, spacy={}, lang='en'):
        SentScores = []
        ReviewlstHeadOutput = []
        ReviewlstModOutput = []
        ReviewlstNegOutput = []
        ReviewlstSentimentOutput = []

        lstScores = []
        lstTokens = []
        lstModifiers = []

        if self.config['PARSER_TYPE'] == 'STANZA':
            Sentences = self.createDicStanza(review)
        elif self.config['PARSER_TYPE'] == 'MODEL':
            Sentences = []
            if(isinstance(review, list)):
                review = review[0]
                spacy = spacy[0]

            #reviews = DataPreprocessing.tokenize_review_into_sentences_nltk(review)
            if self.config['SPLIT_REVIEW'] == 'True':
                reviews = DataPreprocessing.tokenize_review_into_sentences_nltk(review)
            else:
                reviews = [review]
                spacy = [spacy]

            conllu_trees = []
            for idx, sent in enumerate(reviews):
                #SentenceDic = self.createDicModel(sent, [spacy[0][idx]], lang) ### Works for Single Page
                SentenceDic = self.createDicModel(sent, [spacy[idx]], lang)
                #SentenceDic = self.createDicModel(sent, [spacy], lang)
                if(isinstance(SentenceDic, tuple)):
                    Sentences.append(SentenceDic[0])
                    print(SentenceDic[0])
                    #Sentences.append(SentenceDic)
                    conllu_trees.append(SentenceDic[1])
                    #conllu_trees.append(SentenceDic)
            self.dec_tree = conllu_trees
        else:
            print('Wrong mode selection.')

        # Check if the Model failed to parse the sentence and returned error
        if(not isinstance(Sentences, list)):
            SentenceScore = -200
            print("Model failed to parse the sentence and returned error")
            return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

        print("review: {}".format(review))
        #print("number of sentences:", len(Sentences))

        for i in range(len(Sentences)): ### loop over sentences
            #print()
            print("************ sentence {} *************".format(i))
            Sentence = self.FilterNonLemmaWords(Sentences[i]) #new 23 May it's filtering sentences that contain words don't have a lemma key
            Sentence = self.CreateDefaultElementType(Sentence)
            Sentence = self.GetElementScore(Sentence)
            Sentence = self.InitializeSentimentScore(Sentence)
            print("all words: {}".format(self.GetSentenceWords(Sentence)))
            print("sentiment words: {}".format(["{}={}".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]))
            print("sentiment words: {}".format(["[{},{}]".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]))
            sentimentWords = ["[{},{}]".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]
            sentimentWords = [[word["text"], float(word["elementScore"])] for word in Sentence if word["elementScore"] != "none"]
            #print(sentimentWords)
            #SentenceScore= calcSentenceScore(Sentence)
            SentenceScore, lstHeadOutput, lstModOutput, lstNegOutput = self.calcSentenceScore(Sentence)
            #print("lstHeadOutput: {}".format(lstHeadOutput))
            SentScores.append(SentenceScore)
            ReviewlstHeadOutput = ReviewlstHeadOutput + lstHeadOutput
            ReviewlstNegOutput = ReviewlstNegOutput + lstNegOutput
            ReviewlstModOutput = ReviewlstModOutput + lstModOutput
            ReviewlstSentimentOutput = ReviewlstSentimentOutput + sentimentWords
        print("***********************************************")
        print("SenitmentWordslist: ", ReviewlstSentimentOutput)
        print("Headlist: ", ReviewlstHeadOutput)
        print("Negationlist: ", ReviewlstNegOutput)
        print("Modifierlist: ", ReviewlstModOutput)
        print("Sentence scores: ", SentScores)

        if len(SentScores) > 0 and any([score != 'none' for score in SentScores]):
            SentScores = [score for score in SentScores if score != 'none']
            reviewScore = np.mean([float(score) for score in SentScores])
        else:
            reviewScore = -200
        print("The Review Score is ", reviewScore)

        # Iterate over each item in the sentiment words list (ReviewlstHeadOutput) to prepare the lstTokens and lstScores
        for item in ReviewlstSentimentOutput:
            if len(item) > 0:
                lstTokens.append(item[0])
                lstScores.append(item[1]) # modified/final sentiment socore of the token

        # Iterate over each item in the negation words list (ReviewlstNegOutput) to prepare the lstTokens and lstScores
        for item in ReviewlstNegOutput:
            if len(item) > 0:
                lstTokens.append(item[0])
                lstScores.append(item[1]) # modified/final sentiment socore of the token

        for item in ReviewlstModOutput:
            if len(item) > 0:
                double_modifiers = item[0].split(",")
                
                # Deal with double modifiers
                if(len(double_modifiers) > 1):
                    lstModifiers.append(double_modifiers[0].strip())
                    lstModifiers.append(double_modifiers[1].strip())
                    lstTokens.append(double_modifiers[0].strip())
                    lstTokens.append(double_modifiers[1].strip())
                    lstScores.append(item[1])
                    lstScores.append(item[1])
                    #print("double_modifiers", len(double_modifiers))
                else: # Deal with Single Modifiers
                    lstModifiers.append(item[0])
                    lstTokens.append(item[0])
                    lstScores.append(item[1]) # modifier factor applied on the token
        #return reviewScore
        return {"score": reviewScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

    ## function 2 to calculate the sentence score based on the maxima values.
    def calcSentenceScoreMaxima(self, sent, spacy={}, lang='en'):
        lstScores       = []
        lstTokens       = []
        lstModifiers    = []
        
        dicts = {}

        if self.config['PARSER_TYPE'] == 'STANZA':
            dicts = self.createDicStanza(sent)
        elif self.config['PARSER_TYPE'] == 'MODEL':
            dicts = self.createDicModel(sent, spacy, lang)
        else:
            print('Wrong mode selection.')

        #dicts = self.createDicModel(sent, spacy, lang)

        # Check if the Model failed to parse the sentence and returned error
        if(not isinstance(dicts, list)):
            SentenceScore = -200
            return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

        for dicVal in dicts:
            
            #dctChild, dctParent, dctSibling, dctBeforeToken = getChildParentDicts(dicVal)
            
            dic = {}

            dic = self.step1fn(dicVal)

            dctChild, dctParent, dctSibling, dctBeforeToken = self.getChildParentDicts(dic)
            
            branchheadIds = self.calcbranchorder(dctChild)

            if branchheadIds != [0]:
                for headId in branchheadIds:
                    dct = self.calcbranch(dicVal, headId, dctChild[headId])

                    ### Step 4: collect the scores of branch head nodes
                    if "elementType" in dct[headId - 1].keys() and dct[headId - 1]["elementType"] == 'cnt':
                        lstScores.append(round(dct[headId-1]["elementScore"], 2))
                        lstTokens.append(dct[headId-1]["text"])

                    ### Step 4b: collect the scores of branch child nodes, updated
                    for childId in dctChild[headId]:
                        if "elementType" in dct[childId - 1].keys() and dct[childId - 1]["elementType"] == 'cnt':
                            lstScores.append(round(dct[headId-1]["elementScore"], 2))
                            lstTokens.append(dct[headId-1]["text"])

                    #### Prepare MODIFIERS list (Star) *******************************
                    # Iterate through each dictionary in the list
                    for list_item in dct:
                        # Check if the dictionary contains the key 'modifiers_in_sentence'
                        if 'modifiers_in_sentence' in list_item:
                            if(list_item['modifiers_in_sentence'] not in lstModifiers):
                                # If the key is present, append its value to the list
                                lstModifiers.append(list_item['modifiers_in_sentence'])
                    print("lstModifiers", lstModifiers)
                    #### Prepare MODIFIERS list (End) *******************************
            else:
                headId = 1
                dct = self.calcbranch(dicVal, headId, [])

                ### Step 4: collect the scores of branch head nodes
                if "elementType" in dct[headId - 1].keys() and dct[headId - 1]["elementType"] == 'cnt':
                    lstScores.append(round(dct[headId-1]["elementScore"], 2))
                    lstTokens.append(dct[headId-1]["text"])

                ### Step 4b: collect the scores of branch child nodes, updated
                for childId in dctChild[headId]:
                    if "elementType" in dct[childId - 1].keys() and dct[childId - 1]["elementType"] == 'cnt':
                        lstScores.append(round(dct[headId-1]["elementScore"], 2))
                        lstTokens.append(dct[headId-1]["text"])

                #### Prepare MODIFIERS list (Star) *******************************
                # Iterate through each dictionary in the list
                for list_item in dct:
                    # Check if the dictionary contains the key 'modifiers_in_sentence'
                    if 'modifiers_in_sentence' in list_item:
                        if(list_item['modifiers_in_sentence'] not in lstModifiers):
                            # If the key is present, append its value to the list
                            lstModifiers.append(list_item['modifiers_in_sentence'])
                print("lstModifiers", lstModifiers)
                #### Prepare MODIFIERS list (End) *******************************

        if len(lstScores) > 0:
            NumPosScores = sum([1 for score in lstScores if score > 0])
            NumNegScores = sum([1 for score in lstScores if score < 0])

            for x in lstScores:
                if NumPosScores > NumNegScores:
                    SentenceScore = max(lstScores)
                elif NumPosScores < NumNegScores:
                    SentenceScore = min(lstScores)
                else:
                    SentenceScore = mean([float(score) for score in lstScores])
        else:
            SentenceScore = -200
        
        return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}
    
    def scaleB(self, inputVal):
        return ((inputVal+5)/10)*0.50 -0.25

    def normalize_value(self, value, min_in=-5.0, max_in=5.0):
        # Define the range of input values
        min_in = -5
        max_in = 5
        
        # Define the range of output values
        min_out = 1
        max_out = 5
        
        # Normalize the value
        normalized_value = ((value - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
        
        return round(normalized_value)

    def get_default_polarity_label(self):
        default_label = self.config['DEFAULT_POLARITY_LABEL']
        return default_label.capitalize()

    def system_polarity_label(self, score):
        if not isinstance(score, (int, float)):
            #return 'Positive'  # Default "Positive"
            return self.get_default_polarity_label() # Default "Positive"
        polarity_label = ''
        if score > 3.0:
            polarity_label = "Positive"
        elif 0.00 <= score <= 2.00:
            polarity_label = "Negative"
        elif 2.0 < score <= 3.00:
            polarity_label = "Neutral"
        
        return polarity_label

    def modify_conllu_data(self, conllu_data, tokens_list, scores_list, modifiers_list):
        # List negations
        lstNeg = self.lstNeg

        # List Adversatives
        lstAdversatives = ['but', 'although', 'however', 'nevertheless', 'nonetheless', 'nothing']

        # Split the conllu data into lines
        lines = conllu_data.split('\n')
        modified_lines = []
        
        # Initialize the text variable
        text_var = ""
        
        # Iterate over each line starting from the second line
        for line in lines[0:]:
            # Skip empty lines
            if not line.strip():
                modified_lines.append(line)
                continue
            
            # Split the line into fields
            fields = line.split('\t')
            token = fields[1]  # Token is the 2nd field
            
            # Add the 2nd word from each line to the text variable
            text_var += token + " "
            
            # Check if the token is in the tokens list
            if token in tokens_list:
                # Get the corresponding score index
                score_index = tokens_list.index(token)
                score = scores_list[score_index]
                score = round(score, 2)
                
                # Update the last word in the row based on score
                # Available Colors : red, pink, purple, deeppurple, indigo, blue, lightblue, cyan, teal, green, lightgreen, lime, yellow, amber, orange, deeporange, brown, grey, bluegrey
                if score > 0:
                    fields[-1] = "highlight=green"
                    fields[2] = str(score)
                elif score < 0:
                    fields[-1] = "highlight=red"
                    fields[2] = str(score)
            
            # Check if the token is present in the modifiers list
            if token in modifiers_list:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=orange"

            # Check if the token is present in the negations list
            if token in lstNeg:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=brown"

            # Check if the token is present in the adversatives list
            if token in lstAdversatives:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=blue"
            
            # Join the modified fields and add the line to the modified lines list
            modified_line = '\t'.join(fields)
            modified_lines.append(modified_line)
            
        # Insert the concatenated tokens as a comment
        modified_lines.insert(0, f"# text = {text_var}")
        
        # Join the modified lines to form the modified conllu data
        modified_conllu_data = '\n'.join(modified_lines)
    
        return modified_conllu_data

class SpanishSentimentAnalysis:

    _instance = None
   
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(cls, "Object Created!")

        return cls._instance
    
    # List of Negataions
    lstNeg = ['nadie','tampoco','no','ez','non','no','nunca','não','nada','ni','ningun','ninguno','ninguna','nisiquiera', "sin"]

    mode = 'Model'
    dAdj = {}
    dAdv = {}
    dNoun = {}
    dVerb = {}
    dInt = {}

    device = ''
    tokenizer = ''
    model = ''
    dec_tree = ''
    config = ''
    nlp = None

    sentdicts = {"ADV": dAdv, "ADJ": dAdj, "NOUN": dNoun, "VERB": dVerb}

    def __init__(self):
        # Load applicataion config file
        self.config = dotenv_values("config.env")

        # Stanza Nlp
        #stanza.download('en')
        #self.nlp = stanza.Pipeline('en')

        if(self.config['DEFAULT_DIC'] == 'DIC1'): # SO-CAL Dictionary
            self.dAdj = self.populateDictComp(self.config['SPANISH_DIC1_DADJ'])
            self.dAdv = self.populateDictComp(self.config['SPANISH_DIC1_DADV'])
            self.dNoun = self.populateDictComp(self.config['SPANISH_DIC1_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['SPANISH_DIC1_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC2'): # Merged (SO-CAL + VADER)
            self.dAdj = self.populateDictComp(self.config['SPANISH_DIC2_DADJ'])
            self.dAdv = self.populateDictComp(self.config['SPANISH_DIC2_DADV'])
            self.dNoun = self.populateDictComp(self.config['SPANISH_DIC2_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['SPANISH_DIC2_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC3'): # Rest-Mex Only
            self.dAdj = self.populateDictComp(self.config['SPANISH_DIC3_DADJ'])
            self.dAdv = self.populateDictComp(self.config['SPANISH_DIC3_DADV'])
            self.dNoun = self.populateDictComp(self.config['SPANISH_DIC3_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['SPANISH_DIC3_DVERB'])
        elif(self.config['DEFAULT_DIC'] == 'DIC4'): # Merged (Rest-Mex + SO-CAL + VADER)
            self.dAdj = self.populateDictComp(self.config['SPANISH_DIC4_DADJ'])
            self.dAdv = self.populateDictComp(self.config['SPANISH_DIC4_DADV'])
            self.dNoun = self.populateDictComp(self.config['SPANISH_DIC4_DNOUN'])
            self.dVerb = self.populateDictComp(self.config['SPANISH_DIC4_DVERB'])

        # Intensifiers Dictionary
        self.dInt = self.populateDictComp(self.config['SPANISH_INTENSIFIERS'])

        # Manually Fixe Errors in Sentiment Dictionaries
        #dNoun['servicio'] = 2

        # Extra Patch - Fix Issue Rules Overwritten
        dictionaries = [self.dAdj, self.dNoun, self.dVerb, self.dAdv]
        # Iterate through each dictionary
        for d in dictionaries:
            # Iterate through the keys of dInt
            for key in self.dInt.keys():
                # Check if the key exists in the current dictionary
                if key in d:
                    # Delete the entry from the current dictionary
                    del d[key]

        self.sentdicts = {"ADV": self.dAdv, "ADJ": self.dAdj, "NOUN": self.dNoun, "VERB": self.dVerb}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("dccuchile/distilbert-base-spanish-uncased")
        self.model = AutoModelForTokenClassification.from_pretrained(self.config['SPANISH_PARSER_PATH'])
    
    def tokenize_review_into_sentences(self, text_data, lang='es'):
        response = []
        try:
            spacy.prefer_gpu()
            if(lang == 'en'): 
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            elif(lang == 'es'): 
                nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            nlp.add_pipe("sentencizer")
            nlp.max_length = 1500000000

            doc = nlp(text_data)
            sentences = [sent.text for sent in doc.sents]
        except:
            return []
        return sentences

    #Process the dictionaries
    def populateDictComp(self, fileLocation):
        dic = {}
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(fileLocation, 'r', encoding=encoding) as file:
                    for line in file:
                        try:
                            #(key,value) = line.split()
                            #dct[key]=float(value.strip())
                            word, value = line.strip().split('\t')
                            dic[word] = float(value)
                        except ValueError:
                            pass
                            #print("Error parsing line:", line)
            except UnicodeDecodeError:
                #pass
                print(f"Failed to decode using {encoding} encoding")
                print("Failed to decode the file using any of the specified encodings")
        return dic

    def populateDict(self, fileLocation):
        dic = {}
        f = open(fileLocation, encoding='utf-8')
        for line in f:
            words = line.split()
            if(len(words)>2):
                keys = words[:-1]
                key = '-'.join(keys)
                value = words[-1]
                dic[key]=float(value.strip())
            else:
                (key,value) = line.split()
                dic[key] = float(value.strip())
        return dic

    def createDicStanza(self, sent):
        doc = self.nlp(sent)
        dicts = doc.to_dict()
        return dicts

    def createDicModel(self, text, spacy_datas, lang='es'):
        try:
            spacy_lemmas = {}
            dec_tree = ''

            for spacy_data in spacy_datas:
                tokens_with_spacy = spacy_data['tokens']
                if tokens_with_spacy[-1] == '\n':
                    tokens_with_spacy = tokens_with_spacy[:-1]
                
                tokenized_text = pre_tokenized_text(tokens_with_spacy)
                #print("tokenized_text ", tokenized_text)
                # Example pre-tokenized text/data
                #tokenized_text = ["[CLS]", 'I', 'do', "n't", 'eat', 'very', 'much', 'pieces', 'of', 'chocolate', '.', "[SEP]"]

                # Convert tokens to input IDs using the tokenizer
                #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                #print("input_ids ", input_ids)

                # Create input tensors
                input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add a batch dimension
                #print(input_ids)
                #print("input_ids ", input_ids)
                
                with torch.no_grad():
                    #logits = model(**inputs).logits
                    logits = self.model(input_ids).logits

                    predictions = torch.argmax(logits, dim=2)
                    # Model's predictions/rules
                    predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0][1:-1]]

                words = tokens_with_spacy
                postags = spacy_data['postags']
                #print("postags ", postags)
                if postags[-1] == 'SPACE':
                    postags = postags[:-1]
                #print(postags)
                labels = []
                for p in predicted_token_class:
                    if(p != '[CLS]' and p != '[SEP]'):
                        labels.append(D_Label.from_string(p, sep="_"))
                #print("labels ", labels)
                lin_tree = LinearizedTree(words=words, postags=postags,
                            additional_feats=[], labels=labels, n_feats=0)
                #print("lin_tree ", lin_tree)
                #encoder = D_PosBasedEncoding(separator="_")
                encoder = D_NaiveRelativeEncoding(separator="_", hang_from_root=True)
                #encoder = D_NaiveAbsoluteEncoding(separator="_")
                dec_tree = encoder.decode(lin_tree)
                dec_tree.postprocess_tree(search_root_strat=D_ROOT_HEAD, allow_multi_roots=False)
                dec_tree = str(dec_tree)
                #print("dec_tree", dec_tree)
                
                spacy_lemmas = spacy_data['lemmas']
            return tree2dic_optim(dec_tree, spacy_lemmas), dec_tree
        except Exception as e:
            print("Model Inference Error: " + str(e))

    def FilterNonLemmaWords(self, Sentence): # this function deletes problematic lemmas like Spanish: hacerlo
        FilteredSentence = [word for word in Sentence if 'lemma' in word.keys()]
        return (FilteredSentence)
    
    def CreateDefaultElementType(self, Sentence): # assigns to each lemma a default element type 'ord'
        for i in range(len(Sentence)):
            Sentence[i]['elementType'] = 'ord'
        return (Sentence)

    # Set element score from Sentiment dictionaries
    def GetElementScore(self, Sentence):
        for i in range(len(Sentence)):
            lem = Sentence[i]['lemma']
            upos = Sentence[i]['upos']

            if upos == 'ADJ' and lem in self.dAdj.keys():
                Sentence[i]['elementScore'] = self.dAdj[lem]
            elif upos == 'ADV' and lem in self.dAdv.keys():
                Sentence[i]['elementScore'] = self.dAdv[lem]
            elif upos == 'VERB' and lem in self.dVerb.keys():
                Sentence[i]['elementScore'] = self.dVerb[lem]
            elif upos == 'NOUN' and lem in self.dNoun.keys():
                Sentence[i]['elementScore'] = self.dNoun[lem]
            else:
                Sentence[i]['elementScore'] = 'none'
        return Sentence

    def InitializeSentimentScore(self, Sentence): # default is a Sentiment Score
        for i in range(len(Sentence)):
            Sentence[i]['SentimentScore'] = 'none'
        return Sentence

    def GetSentenceWords(self, Sentence): # important for the demo
        #Sentence is a list of word dictionaries
        lstwords = []
        for word in Sentence:
            lstwords.append(word["text"])
        return (lstwords)

    def getChildParentDicts(self, dct):
        dctChild       = {} ## for now, this dictionary is important
        dctParent      = {}
        dctBeforeToken = {}

        for dicVal in dct:
            #print(dicVal)
            elementID     = dicVal['id']
            elementHead   = dicVal['head']
            elementLemma  = dicVal['lemma']

            if(elementID != 1 ):
                eleBeforeDic = dct[int(elementID)-2]
                elementBeforLemma = eleBeforeDic['lemma']
                #print(elementID, elementLemma, eleBeforeDic, eleBeforeDic['lemma'], elementBeforLemma)
            else:
                elementBeforLemma = "-"

            # dic parent is done
            dctParent[elementID] = elementHead
            dctBeforeToken[elementLemma] = elementBeforLemma

            if elementHead not in dctChild.keys():
                dctChild[elementHead] = []

            # Adding to the parent node (Elements and ElementsLemma)
            dctChild[elementHead].append(elementID)

        # working on Sibling dictionary
        dctSibling = {}

        # Giviing some default values
        for key in dctParent.keys():
            dctSibling[key] = []
        for key in dctChild.keys():
            childList = dctChild[key]
            #print(childList)
            for chld in childList:
                #print("Child is: ", chld)
                dctSibling[chld] = [x for x in childList if x != chld]
        return dctChild, dctParent, dctSibling, dctBeforeToken

    def calcbranchorder(self, dctChild):
        #print("dctChild.keys are", list(dctChild.keys()))
        if list(dctChild.keys()) == [0]:
            calcheads=[0]
        else:
            allheads = list(dctChild.keys())
            calcheads = []
            #children = dctChild[0]
            children = dctChild.get(0)  # Returns None if key doesn't exist
            children = children or []   # Assuming you want an empty list if children is None
            nextchildren = []
            numchildren = len(children)
            #print("numchildren is", numchildren)
            while numchildren > 0:
                nextchildren = []
                #print('numchildren = ', numchildren)
                for child in children:
                    #print('child = ', child)
                    if child in allheads:
                        #print('headchild = ', child)
                        calcheads.append(child)
                        nextchildren = nextchildren + dctChild[child]
                        #print('calcheads = ', calcheads)
                        #print('nextchildren = ', nextchildren)
                children = nextchildren
                numchildren = len(children)
            calcheads.reverse()
        #print('branch order = ', calcheads)
        return calcheads

    def calcbranch(self, dct, headId, childIds):
        dInt        = self.dInt
        lstNeg      = self.lstNeg
        sentdicts   = self.sentdicts

        #dct is a list of word dictionaries like Sentence
        lstchildren = [dct[childId-1]["text"] + "({},{})".format(childId, dct[childId-1]["deprel"]) for childId in childIds]
        #lstchildren=[dct[childId-1]["deprel"] + "({})".format(childId) for childId in childIds]
        #print("head: {}({}), children: {}".format(dct[headId-1]["text"], headId, lstchildren) )
        
        ####finding relevant elements in each branch
        a           = 0
        b           = 0
        c           = 0 # new 3.6.
        neg         = 0
        p           = 0
        pvalue      = 0.25
        negvalue    = -4
        Negstat     = False
        fmodstat    = False
        lexmodstat  = False
        conjmodstat = False # new 3.6.
        lexstat     = False
        lstcntchildren  = []
        lstmodification = []
        lstnegation     = []
        PrintList       = []
        #####create default outputs
        HeadOutput      = []
        NegOutput       = []
        ModOutput       = []

    #### starting with children of the branch
        PrintList.append('Number of Children: {}'.format(len(childIds)))
        for i in range(len(childIds)):
        #for childId in childIds:###
            #print("childid=", childId)
            childId = childIds[i]
            lemma = dct[childId-1]["lemma"]
            text = dct[childId-1]["text"]
            pos = dct[childId-1]["upos"]#
            deprel = dct[childId-1]["deprel"] # head, amod, nmod, mod in deprel
            elementScore = dct[childId-1]["elementScore"]
            if dct[childId-1]["SentimentScore"] != 'none':
                SentimentScore = dct[childId-1]["SentimentScore"]
            else:
                SentimentScore = elementScore
            #lstchildren[i]=lstchildren[i] + "({})".format(deprel)

            if lemma in lstNeg or text in lstNeg: # find negation
                ### Child cannot be a head. ###
                Negstat = True
                neg = negvalue
                lstchildren[i] = lstchildren[i] + "({})".format("N")
                dct[childId-1]['elementType'] = 'neg'
                #print("negation token is: ", text, "negation childid is: ", childId, "the negation value is: ", neg) #information for the demo
                #PrintList.append("Child{}: '{}'({}), Neg={}".format(i, text, childId, neg)) # important for the demo
                NegOutput = [text,neg]
            # elif lemma == 'uno' and dct[childId]["lemma"] == 'poco':
            #     fmodstat = True
            #     b=-0.3
            elif text == 'un' and pos=="DET" and deprel == 'advmod': # 31.5 accounting for "un poco as in un poco triste or ofrece un poco."
                fmodstat = True
                b = -0.3
                lstchildren[i] = lstchildren[i] + "(un poco={})".format(b)
                ModOutput = [text,b]
            elif lemma in dInt.keys() and ('mod' in deprel or 'det' in deprel or 'obj' in deprel):# new added 26 May,  find intensification
            #elif lemma in dInt.keys() and ('mod' in deprel or 'det' in deprel):  # update on 3.6.
                ### Child cannot be a head. ###
                fmodstat = True
                PrevExists = (childId-2) >= 0
                NextExists = childId <= len(dct)-1
                if PrevExists:
                    PrevIsInt = dct[childId - 2]["lemma"] in dInt.keys()
                    PrevIsNeg = dct[childId - 2]["lemma"] in lstNeg
                    PrevIsHead = dct[childId - 2]["head"] == dct[childId-1]["id"]
                else:
                    PrevIsInt = False
                    PrevIsNeg = False
                    PrevIsHead = False
                if NextExists:
                    NextIsInt = dct[childId]["lemma"] in dInt.keys()
                else:
                    NextIsInt = False
                if PrevIsInt == False and NextIsInt == False: # single intensifier found, e.g. a VERY good service
                    if PrevIsNeg and PrevIsHead: # looking for head negation of a single intensifier, e.g. not many problems (Stanza analyses not as a head of many)
                        if dInt[lemma] > 0:
                            b = -abs(dInt[lemma])
                        if dInt[lemma] < 0:
                            b = abs(dInt[lemma])
                    else:
                        b = dInt[lemma]
                    #print("{}single modifier={}({}), b={}".format(indent, text, childId, b)) ###new 23.5
                    PrintList.append("single modifier={}({}), b={}".format(text, childId, b)) ###new 23.5
                    PrintList.append("Child{}: '{}'({}), 1b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(1b={})".format(b)
                    ModOutput = [text,b]
                elif PrevIsInt: #double modifier found, e.g. little bit, very small, etc.
                    lemma2 = dct[childId - 2]["lemma"]
                    fmodscore = dInt[lemma]*(1+dInt[lemma2]) # a compound b or 2b
                    b = fmodscore
                    #print("double modifier is:", text, "childid is: ", childId, "the value of double modifier is: ",  b) ###new 22.5
                    PrintList.append("double modifier={}({}), b={}".format(text, childId, b)) ###new 23.5, double modifier
                    PrintList.append("Child{}: '{}'({}), 2b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(2b={})".format(b)
                    ModOutput = ["{}, {}".format(dct[childId - 2]["text"], text), b]
                else:
                    PrintList.append("Child{}: '{}'({}), first part of 2b={}".format(i, text, childId, b))
                    lstchildren[i] = lstchildren[i] + "(2b={})".format(b)
            #elif dct[childId-1]['feats']=='Degree=Sup': #needs to be implemented
                #lexmodstat=True
                #lexmodscore= elementScore*(1+0.7)
                #c=lexmodscore
                #print("child is a lexical modifier with a superlative morpheme")
            #elif pos in sentdicts.keys(): #only processes it if lexmodstat is not satisfied???
            #elif True:
            elif (lemma in sentdicts.keys() or SentimentScore!='none') and 'cl' not in deprel and 'conj' not in deprel: #new update on 3.6= children are neither adj, verbs, nouns or adverbs, but pronouns with a SentimenSscore
                lexmodstat=True
                a=SentimentScore
                lstchildren[i]=lstchildren[i] + "(a={})".format(SentimentScore)
            elif (lemma in sentdicts.keys() or SentimentScore!='none') and ('cl' in deprel or 'conj' in deprel):  # new update on 3.6.
                conjmodstat=True
                c=SentimentScore
                lstchildren[i]=lstchildren[i] + "(c={})".format(SentimentScore)
            elif lemma in ['pero', 'obstante', 'sino', 'aunque', 'malgrado']:
                ### Child cannot be a head. ###
                p = pvalue
                lstchildren[i] = lstchildren[i] + "(p={})".format(p)
                #print("but is present and has the value of: ", p) # for the demo
                #PrintList.append("{} is present, p=".format(lemma, p)) # for the demo
                PrintList.append("Child{}: '{}'({}), p={}".format(i, text, childId, p))
            else:
                lstchildren[i]=lstchildren[i] + "({}={})".format(pos,SentimentScore)
                PrintList.append("Child{}: '{}'({})".format(i, text, childId))
                #lstchildren[i] = lstchildren[i] + "({}={})".format(pos,SentimentScore)
                #PrintList.append("Child{}: '{}'({})".format(i, text, childId))
            #print("child: ", childId, text, elementScore)
        PrintList.append("Children-derived score elements: a={}, b={}, neg={}, p={}".format(a, b, neg, p)) # relevant for demo
    
        ######################################################################################################################################
        ####continuing with parents/heads of the branch ###
        lemma = dct[headId-1]["lemma"]
        pos = dct[headId-1]["upos"]
        deprel = dct[headId-1]["deprel"]
        text = dct[headId-1]["text"]
        #headmodstat = "mod" in deprel or 'conj' in deprel
        headmodstat = "mod" in deprel
        #headmodstat = "mod" in deprel or 'DET' in 'upos' #new
        headscore = "none"
        headsentimentscore = 'none'
        childp = p
        head_a = 0
        head_b = 0
        head_neg = 0
        head_p = 0
        calc_a, calc_b, calc_neg, calc_p = 0, 0, 0, 0
        words_to_match = ["barbaridad", "par", "montón"]
        lstheads=[]

        if lemma == "nada":
            head_neg = negvalue
            #headsentimentscore = (a * (1+head_b)  + (np.sign(a*(1+head_b)))* head_neg)*(1+p) # new 25.5
            headsentimentscore=(a * (1+b)  + (np.sign(a*(1+b)))* head_neg)*(1+p) # new update on 3rd of June
            dct[headId-1]["SentimentScore"] = headsentimentscore #new
            PrintList.append("Head: '{}'({}), 'nada'={}, SentimentScore={}".format(text, headId, head_neg, headsentimentscore))
            lstheads.append([text,head_neg, headsentimentscore])
            #calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
    
        # Example condition
        elif lemma in words_to_match and pos =="NOUN":
        #elif lemma in dInt.keys(): #new 14.5. accounting for intensifiers as heads as in "a bunch of problems" or 'a bit of heaven'
            fmodscore = dInt[lemma]
            head_b = fmodscore
            fmodstat = True
            if True: #all([(type(x) is int) or (type(x) is float) for x in [a, head_b, p, neg]]):
                headsentimentscore = (a * (1+head_b)  + (np.sign(a*(1+head_b)))* neg)*(1+p) # new 25.5
                #headsentimentscore=(childsentimentscore * (1+head_b)  + (np.sign(childsentimentscore*(1+head_b)))* neg)*(1+p)# new 26.5
                dct[headId-1]["SentimentScore"] = headsentimentscore #new
                PrintList.append("Head: '{}'({}), head intensifier, b={}, SentimentScore={}".format(text, headId, head_b, headsentimentscore))
                HeadOutput = [text, a, headsentimentscore]
                #print("HeadOutput", HeadOutput)
                #calc_a, calc_b, calc_neg, calc_p = a, head_b, neg, p
            else:
                PrintList.append("Head: '{}'({}), a, b, p or neg is not a number, SentimentScore={}".format(text, headId, headsentimentscore))
                calc_a, calc_b, calc_neg, calc_p = a, head_b, neg, p
        elif lexmodstat: # if one of the children is a lexical head, then do the calculation of headsentimentscore.
            if True: #all([(type(x) is int) or (type(x) is float) for x in [a, b, p, neg]]):
                calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                headsentimentscore = (a * (1+b)  + (np.sign(a*(1+b)))* neg)*(1+p)#25.5.
                #headsentimentscore=(childsentimentscore * (1+b)  + (np.sign(childsentimentscore*(1+b)))* neg)*(1+p) # 26.5
                dct[headId-1]["SentimentScore"] = headsentimentscore # the head gets the polarity score of the child
                PrintList.append("Head: '{}'({}), child is a sentiment word, a={}, SentimentScore={}".format(text, headId, a, headsentimentscore)) # 25.5.
                #PrintList.append("Head inheriting Childfeatures: '{}'({}), childsentimentscore={}, SentimentScore={}".format(text, headId, childsentimentscore, headsentimentscore)) # 26.5.
                #calc_a, calc_b, calc_neg, calc_p = a, b, neg, p
                HeadOutput = [text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
            else:
                PrintList.append("Head: '{}'({}), a, b, p or neg is not a number, SentimentScore={}".format(text, headId, a, headsentimentscore))
        elif pos in sentdicts.keys() and conjmodstat!=True: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
            dsent= sentdicts[pos]
            if lemma in dsent.keys():
                headscore=dsent[lemma]
                head_a=dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                headsentimentscore=(head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)
                PrintList.append("Head: '{}'({}), head is a sentiment word, head_a={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput=[text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"]= headsentimentscore
                #childp=0
        elif pos in sentdicts.keys() and conjmodstat: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
            dsent = sentdicts[pos]
            if lemma in dsent.keys(): #or SentimentScore!='none':
                headscore = dsent[lemma]
                head_a = dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                headsentimentscore = np.mean([((head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)), c])
                PrintList.append("Head: '{}'({}), head is a sentiment word with a conjunction head, head_a={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput =[text, a, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"] = headsentimentscore
                #childp=0
            else: #only processes if lexmodstat (=child has a sentimentscore) is not satisfied
                headsentimentscore = c
                #headscore=dsent[lemma]
                #head_a=dsent[lemma]
                #calc_a, calc_b, calc_neg, calc_p = head_a, b, neg, p
                #headsentimentscore=np.mean([((head_a * (1+b)  + (np.sign(head_a*(1+b)))* neg)*(1+p)), c])
                PrintList.append("Head: '{}'({}), head has no sentiment word with a conjunction head, c={}, SentimentScore={}".format(text, headId, head_a, headsentimentscore))
                HeadOutput = [text, c, headsentimentscore]
                PrintList.append("HeadOutput {}".format(HeadOutput))
                dct[headId-1]["SentimentScore"] = headsentimentscore
        else:
            PrintList.append("Head: '{}'({}), No calculation possible, SentimentScore={}".format(text, headId, headsentimentscore))
        #PrintList.append("Head-derived score elements: head_a={}, head_b={}, head_neg={}, head_p={}".format(head_a, head_b, head_neg, head_p))
        #PrintList.append("Calculation score elements: calc_a={}, calc_b={}, calc_neg={}, calc_p={}".format(calc_a, calc_b, calc_neg, calc_p))
        #PrintList.append('Result of Calculation:')
        #PrintList.append("head: {}({}), children: {}".format(dct[headId-1]["text"], headId, lstchildren))
        #print("list of heads:", lstheads)
        #print("list of negation:", lstnegation)
        #print("list of modification:", lstmodification)
        #print("ListChildren is:", lstchildren) #new 27.5.
        #print("HeadOutput", HeadOutput)
        #indent = "      " #indent each statement by this much.
        #for x in PrintList:
            #print(indent + x)
        return dct, HeadOutput, NegOutput, ModOutput
    
    def calcSentenceScore(self, dct): # dct= Sentence which is a list of word dictionaries and the function takes only one sentence
        lstScores = []
        lstHeadOutput = []
        lstModOutput = []
        lstNegOutput = []
        dctChild, dctParent, dctSibling, dctBeforeToken = self.getChildParentDicts(dct)
        #print(dctChild)

        ###Step 2 figure out order of nodes
        branchheadIds = self.calcbranchorder(dctChild)
        #print("branchheadIds: {}".format(branchheadIds))
        #topheadid = branchheadIds[-1] # new updated on 22nd of May.
        topheadid = None if not branchheadIds else branchheadIds[-1]


        ## Step 3 looping over nodes
        #print('Number of branches: {}'.format(len(branchheadIds)))
        if branchheadIds != [0] and len(branchheadIds) > 0: # calculate branches that have more than one word.
            for BranchIndex in range(len(branchheadIds)): #looping over nodes
                headId = branchheadIds[BranchIndex]
                #print('Branch{}'.format(BranchIndex))
                #print("Branch{} = head: '{}', children: {}".format(BranchIndex, dct[headId-1]['text'],[dct[Id-1]["text"] for Id in dctChild[headId]]))
                #dct=calcbranch(dct, headId, dctChild[headId])
                dct, HeadOutput, NegOutput, ModOutput = self.calcbranch(dct, headId, dctChild[headId])
                #if
                lstHeadOutput.append(HeadOutput)
                lstModOutput.append(ModOutput)
                lstNegOutput.append(NegOutput)
                
        ###Step 4 collect the scores of branchheadIds
                #if dct[headId-1]["elementScore"]!="none":
                #if dct[headId-1]["SentimentScore"]!="none":
                if headId == topheadid and dct[headId-1]["elementScore"] != "none":
                    #lstScores.append(dct[headId-1]["elementScore"])
                    lstScores.append(dct[headId-1]["SentimentScore"])
                    #print("lstScores", lstScores)
        else: # calculate branches with only one word
            headId = 1 #headId=0, ChildId=[1] dctChildren= {0: [1]}
            #dct=calcbranch(dct, headId, [])
            dct, HeadOutput, NegOutput, ModOutput = self.calcbranch(dct, headId, [])
            lstHeadOutput.append(HeadOutput)
            lstModOutput.append(ModOutput)
            lstNegOutput.append(NegOutput)
            #dct=calcbranch(dct, headId, dctChild[headId]) #new
            #print("elementScore:{}".format(dct[headId - 1]["elementScore"]))
            #print("SentimentScore:{}".format(dct[headId - 1]["SentimentScore"]))
        ###Step 4 collect the scores of branchheadIds
        if dct[headId - 1]["SentimentScore"] != "none": # dct[headId - 1]["elementScore"] does not exist because it is either none nor some score
            #print("elementScore:{}".format(dct[headId - 1]["elementScore"]))
            #print("SentimentScore:{}".format(dct[headId - 1]["SentimentScore"]))
            #if dct[headId - 1]["elementScore"]=="none":
            #lstScores.append(dct[headId - 1]["elementScore"])
            lstScores.append(dct[headId - 1]["SentimentScore"])
            #print("lstScores", lstScores)
        else:
            lstScores.append('none')
        ###Step 5 create a sentence score
        #print("lstScores: {}".format(lstScores))
        if len(lstScores) != 0 and any([score != 'none' for score in lstScores]):
            SentenceScore = np.mean([float(score) for score in lstScores if score != 'none'])
            return SentenceScore, lstHeadOutput, lstModOutput, lstNegOutput
        else:
            return 'none', lstHeadOutput, lstModOutput, lstNegOutput

    def calcReviewScore(self, review, spacy={}, lang='es'):
        #print("SpaCy Data in calcReviewScore: ", spacy)
        SentScores = []
        ReviewlstHeadOutput = []
        ReviewlstModOutput = []
        ReviewlstNegOutput = []
        ReviewlstSentimentOutput = []

        lstScores = []
        lstTokens = []
        lstModifiers = []

        if self.config['PARSER_TYPE'] == 'STANZA':
            Sentences = self.createDicStanza(review)
            #print("STANZA", Sentences)
        elif self.config['PARSER_TYPE'] == 'MODEL':
            Sentences = []
            #print(type(review), review)
            if(isinstance(review, list)):
                review = review[0]
                spacy = spacy[0]

            #reviews = DataPreprocessing.tokenize_review_into_sentences_nltk(review)
            if self.config['SPLIT_REVIEW'] == 'True':
                reviews = DataPreprocessing.tokenize_review_into_sentences_nltk(review, language='spanish')
            else:
                reviews = [review]
                spacy = [spacy]

            conllu_trees = []
            for idx, sent in enumerate(reviews):
                #SentenceDic = self.createDicModel(sent, [spacy[0][idx]], lang) ### Works for Single Page
                SentenceDic = self.createDicModel(sent, [spacy[idx]], lang)
                #print("SentenceDic ", SentenceDic)
                #SentenceDic = self.createDicModel(sent, [spacy], lang)
                if(isinstance(SentenceDic, tuple)):
                    Sentences.append(SentenceDic[0])
                    #Sentences.append(SentenceDic)
                    conllu_trees.append(SentenceDic[1])
                    #conllu_trees.append(SentenceDic)
            self.dec_tree = conllu_trees
        else:
            print('Wrong mode selection.')

        # Check if the Model failed to parse the sentence and returned error
        if(not isinstance(Sentences, list)):
            SentenceScore = -200
            print("Model failed to parse the sentence and returned error")
            return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

        #print("review: {}".format(review))
        #print("number of sentences:", len(Sentences))

        for i in range(len(Sentences)): ### loop over sentences
            #print()
            #print("************ sentence {} *************".format(i))
            Sentence = self.FilterNonLemmaWords(Sentences[i]) #new 23 May it's filtering sentences that contain words don't have a lemma key
            Sentence = self.CreateDefaultElementType(Sentence)
            Sentence = self.GetElementScore(Sentence)
            Sentence = self.InitializeSentimentScore(Sentence)
            #print("all words: {}".format(self.GetSentenceWords(Sentence)))
            #print("sentiment words: {}".format(["{}={}".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]))
            #print("sentiment words: {}".format(["[{},{}]".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]))
            #sentimentWords = ["[{},{}]".format(word["text"], word["elementScore"]) for word in Sentence if word["elementScore"]!="none"]
            sentimentWords = [[word["text"], float(word["elementScore"])] for word in Sentence if word["elementScore"] != "none"]
            #print(sentimentWords)
            #SentenceScore= calcSentenceScore(Sentence)
            SentenceScore, lstHeadOutput, lstModOutput, lstNegOutput = self.calcSentenceScore(Sentence)
            #print("lstHeadOutput: {}".format(lstHeadOutput))
            SentScores.append(SentenceScore)
            ReviewlstHeadOutput = ReviewlstHeadOutput + lstHeadOutput
            ReviewlstNegOutput = ReviewlstNegOutput + lstNegOutput
            ReviewlstModOutput = ReviewlstModOutput + lstModOutput
            ReviewlstSentimentOutput = ReviewlstSentimentOutput + sentimentWords
        #print("***********************************************")
        #print("SenitmentWordslist: ", ReviewlstSentimentOutput)
        #print("Headlist: ", ReviewlstHeadOutput)
        #print("Negationlist: ", ReviewlstNegOutput)
        #print("Modifierlist: ", ReviewlstModOutput)
        #print("Sentence scores: ", SentScores)

        if len(SentScores) > 0 and any([score != 'none' for score in SentScores]):
            SentScores = [score for score in SentScores if score != 'none']
            reviewScore = np.mean([float(score) for score in SentScores])
        else:
            reviewScore = -200
        print("The Review Score is ", reviewScore)

        # Iterate over each item in the sentiment words list (ReviewlstHeadOutput) to prepare the lstTokens and lstScores
        for item in ReviewlstSentimentOutput:
            if len(item) > 0:
                lstTokens.append(item[0])
                lstScores.append(item[1]) # modified/final sentiment socore of the token

        for item in ReviewlstModOutput:
            if len(item) > 0:
                double_modifiers = item[0].split(",")
                
                # Deal with double modifiers
                if(len(double_modifiers) > 1):
                    lstModifiers.append(double_modifiers[0].strip())
                    lstModifiers.append(double_modifiers[1].strip())
                    lstTokens.append(double_modifiers[0].strip())
                    lstTokens.append(double_modifiers[1].strip())
                    lstScores.append(item[1])
                    lstScores.append(item[1])
                    #print("double_modifiers", len(double_modifiers))
                else: # Deal with Single Modifiers
                    lstModifiers.append(item[0])
                    lstTokens.append(item[0])
                    lstScores.append(item[1]) # modifier factor applied on the token
        #return reviewScore
        return {"score": reviewScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

    ## function 2 to calculate the sentence score based on the maxima values.
    def calcSentenceScoreMaxima(self, sent, spacy={}, lang='en'):
        lstScores       = []
        lstTokens       = []
        lstModifiers    = []
        
        dicts = {}

        if self.config['PARSER_TYPE'] == 'STANZA':
            dicts = self.createDicStanza(sent)
        elif self.config['PARSER_TYPE'] == 'MODEL':
            dicts = self.createDicModel(sent, spacy, lang)
        else:
            print('Wrong mode selection.')

        #dicts = self.createDicModel(sent, spacy, lang)

        # Check if the Model failed to parse the sentence and returned error
        if(not isinstance(dicts, list)):
            SentenceScore = -200
            return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}

        for dicVal in dicts:
            
            #dctChild, dctParent, dctSibling, dctBeforeToken = getChildParentDicts(dicVal)
            
            dic = {}

            dic = self.step1fn(dicVal)

            dctChild, dctParent, dctSibling, dctBeforeToken = self.getChildParentDicts(dic)
            
            branchheadIds = self.calcbranchorder(dctChild)

            if branchheadIds != [0]:
                for headId in branchheadIds:
                    dct = self.calcbranch(dicVal, headId, dctChild[headId])

                    ### Step 4: collect the scores of branch head nodes
                    if "elementType" in dct[headId - 1].keys() and dct[headId - 1]["elementType"] == 'cnt':
                        lstScores.append(round(dct[headId-1]["elementScore"], 2))
                        lstTokens.append(dct[headId-1]["text"])

                    ### Step 4b: collect the scores of branch child nodes, updated
                    for childId in dctChild[headId]:
                        if "elementType" in dct[childId - 1].keys() and dct[childId - 1]["elementType"] == 'cnt':
                            lstScores.append(round(dct[headId-1]["elementScore"], 2))
                            lstTokens.append(dct[headId-1]["text"])

                    #### Prepare MODIFIERS list (Star) *******************************
                    # Iterate through each dictionary in the list
                    for list_item in dct:
                        # Check if the dictionary contains the key 'modifiers_in_sentence'
                        if 'modifiers_in_sentence' in list_item:
                            if(list_item['modifiers_in_sentence'] not in lstModifiers):
                                # If the key is present, append its value to the list
                                lstModifiers.append(list_item['modifiers_in_sentence'])
                    print("lstModifiers", lstModifiers)
                    #### Prepare MODIFIERS list (End) *******************************
            else:
                headId = 1
                dct = self.calcbranch(dicVal, headId, [])

                ### Step 4: collect the scores of branch head nodes
                if "elementType" in dct[headId - 1].keys() and dct[headId - 1]["elementType"] == 'cnt':
                    lstScores.append(round(dct[headId-1]["elementScore"], 2))
                    lstTokens.append(dct[headId-1]["text"])

                ### Step 4b: collect the scores of branch child nodes, updated
                for childId in dctChild[headId]:
                    if "elementType" in dct[childId - 1].keys() and dct[childId - 1]["elementType"] == 'cnt':
                        lstScores.append(round(dct[headId-1]["elementScore"], 2))
                        lstTokens.append(dct[headId-1]["text"])

                #### Prepare MODIFIERS list (Star) *******************************
                # Iterate through each dictionary in the list
                for list_item in dct:
                    # Check if the dictionary contains the key 'modifiers_in_sentence'
                    if 'modifiers_in_sentence' in list_item:
                        if(list_item['modifiers_in_sentence'] not in lstModifiers):
                            # If the key is present, append its value to the list
                            lstModifiers.append(list_item['modifiers_in_sentence'])
                print("lstModifiers", lstModifiers)
                #### Prepare MODIFIERS list (End) *******************************

        if len(lstScores) > 0:
            NumPosScores = sum([1 for score in lstScores if score > 0])
            NumNegScores = sum([1 for score in lstScores if score < 0])

            for x in lstScores:
                if NumPosScores > NumNegScores:
                    SentenceScore = max(lstScores)
                elif NumPosScores < NumNegScores:
                    SentenceScore = min(lstScores)
                else:
                    SentenceScore = mean([float(score) for score in lstScores])
        else:
            SentenceScore = -200
        
        return {"score": SentenceScore, "lstScores": lstScores, "lstTokens": lstTokens, "lstModifiers": lstModifiers}
    
    def scaleB(self, inputVal):
        return ((inputVal+5)/10)*0.50 -0.25

    def normalize_value(self, value, min_in=-5.0, max_in=5.0):
        # Define the range of input values
        min_in = -5
        max_in = 5
        
        # Define the range of output values
        min_out = 1
        max_out = 5
        
        # Normalize the value
        normalized_value = ((value - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
        
        return round(normalized_value)

    def get_default_polarity_label(self):
        default_label = self.config['DEFAULT_POLARITY_LABEL']
        return default_label.capitalize()

    def system_polarity_label(self, score):
        if not isinstance(score, (int, float)):
            #return 'Positive'  # Default "Positive"
            return self.get_default_polarity_label() # Default "Positive"
        polarity_label = ''
        if score > 3.0:
            polarity_label = "Positive"
        elif 0.00 <= score <= 2.00:
            polarity_label = "Negative"
        elif 2.0 < score <= 3.00:
            polarity_label = "Neutral"
        
        return polarity_label

    def modify_conllu_data(self, conllu_data, tokens_list, scores_list, modifiers_list):
        # List negations
        lstNeg = self.lstNeg

        # List Adversatives
        lstAdversatives = ['pero', 'sino', 'aunque', 'nonobstante', 'nada']

        # Split the conllu data into lines
        lines = conllu_data.split('\n')
        modified_lines = []
        
        # Initialize the text variable
        text_var = ""
        
        # Iterate over each line starting from the second line
        for line in lines[0:]:
            # Skip empty lines
            if not line.strip():
                modified_lines.append(line)
                continue
            
            # Split the line into fields
            fields = line.split('\t')
            token = fields[1]  # Token is the 2nd field
            
            # Add the 2nd word from each line to the text variable
            text_var += token + " "
            
            # Check if the token is in the tokens list
            if token in tokens_list:
                # Get the corresponding score index
                score_index = tokens_list.index(token)
                score = scores_list[score_index]
                score = round(score, 2)
                
                # Update the last word in the row based on score
                # Available Colors : red, pink, purple, deeppurple, indigo, blue, lightblue, cyan, teal, green, lightgreen, lime, yellow, amber, orange, deeporange, brown, grey, bluegrey
                if score > 0:
                    fields[-1] = "highlight=green"
                    fields[2] = str(score)
                elif score < 0:
                    fields[-1] = "highlight=red"
                    fields[2] = str(score)
            
            # Check if the token is present in the modifiers list
            if token in modifiers_list:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=orange"

            # Check if the token is present in the negations list
            if token in lstNeg:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=brown"

            # Check if the token is present in the adversatives list
            if token in lstAdversatives:
                # If found, apply 'highlight:orange' to the last field
                fields[-1] = "highlight=blue"
            
            # Join the modified fields and add the line to the modified lines list
            modified_line = '\t'.join(fields)
            modified_lines.append(modified_line)
            
        # Insert the concatenated tokens as a comment
        modified_lines.insert(0, f"# text = {text_var}")
        
        # Join the modified lines to form the modified conllu data
        modified_conllu_data = '\n'.join(modified_lines)
    
        return modified_conllu_data