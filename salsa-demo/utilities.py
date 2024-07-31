import warnings
warnings.simplefilter("ignore", UserWarning)

from transformers import AutoTokenizer
import torch
import spacy
import stanza

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            new_labels.append(-100)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["syntax_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
	
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
    
def tokenize_and_align_labels_for_constituency(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["target"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def pre_tokenized_text(words):
    text_tokens = ["[CLS]"]
    for word in words:
        text_tokens.append(word.lower())
    text_tokens.append("[SEP]")
    return text_tokens

def get_tokens_by_spacy(text):
    # Create a pipeline
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "ner"])
    docs = nlp(text)
    tokens = []
    for doc in docs:
        tokens.append(doc.text)
    return tokens

def get_tokens_by_stanza(text):
    # Create a pipeline
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    #nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "ner"])
    docs = nlp(text)
    tokens = []
    for sent in docs.sentences:
        for word in sent.words:
            tokens.append(word.text)
    return tokens 

def write_to_file(content, path):
    output_f = open(path, "a+")
    output_f.write(content)
    #output_f.write("\n")
    #output_f.write("\n")
    output_f.close()

def get_sentence(rows):
    words = [row.split('\t')[1] for row in rows]
    sentence = ' '.join(words)
    return sentence

def get_lemmas_by_spacy(text):
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "ner"])
    #nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])
    # Process the text and get lemmatized output
    tokens = nlp(text)
    lemmas_dic = {}
    for token in tokens:
        lemmas_dic.setdefault(token.text, token.lemma_)
    return lemmas_dic

def get_pos_by_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(text)
    pos_tags = []
    for token in tokens:
        pos_tags.append(token.pos_)
    return pos_tags

def tree2dic(dec_tree):
    rows = dec_tree.splitlines()[:-1] # skip the empty last index
    sentence = get_sentence(rows)
    lemmas = get_lemmas_by_spacy(sentence)
    fields_mappings = []
    
    for row in rows:
        fields = row.split('\t')
        if fields[1].startswith("'") or fields[1].startswith("’"):
            lemma = fields[1][1:].lower()
        else:
            lemma = lemmas[fields[1]]
        dic = {
            'id': int(fields[0]),
            'text': fields[1],
            #'lemma': lemmas[fields[1]],
            'lemma': lemma,
            'upos': fields[3],
            'head': int(fields[6]),
            'deprel': fields[7]
        }
        fields_mappings.append(dic)
    return fields_mappings

### Performance Optimzed functions for model integration
def spacy_helper(text):
    #tokens_with_spacy = get_tokens_by_spacy(text)
    #postags = get_pos_by_spacy(text)
    #lemmas = get_lemmas_by_spacy(sentence)

    # Create a pipeline
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    #nlp = spacy.load("en_core_web_sm", disable=["attribute_ruler", "ner"])
    docs = nlp(text)
    tokens = []
    pos_tags = []
    lemmas_dic = {}

    for doc in docs:
        tokens.append(doc.text)
        pos_tags.append(doc.pos_)
        lemmas_dic.setdefault(doc.text, doc.lemma_)
    return {"tokens": tokens, "pos_tags": pos_tags, "lemmas": lemmas_dic}

def tree2dic_optim(dec_tree, lemmas):
    rows = dec_tree.splitlines()[:-1] # skip the empty last index
    sentence = get_sentence(rows)
    #lemmas = get_lemmas_by_spacy(sentence)
    fields_mappings = []
    
    for row in rows:
        fields = row.split('\t')
        if fields[1].startswith("'") or fields[1].startswith("’"):
            lemma = fields[1][1:].lower()
        else:
            lemma = lemmas[fields[1]]
        dic = {
            'id': int(fields[0]),
            'text': fields[1],
            #'lemma': lemmas[fields[1]],
            'lemma': lemma,
            'upos': fields[3],
            'head': int(fields[6]),
            'deprel': fields[7]
        }
        fields_mappings.append(dic)
    return fields_mappings

### Batch Processing functions for model integration
def batch_spacy_helper(text):
    #tokens_with_spacy = get_tokens_by_spacy(text)
    #postags = get_pos_by_spacy(text)
    #lemmas = get_lemmas_by_spacy(sentence)

    # Create a pipeline
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    #nlp = spacy.load("en_core_web_sm", disable=["attribute_ruler", "ner"])
    docs = nlp(text)
    tokens = []
    pos_tags = []
    lemmas_dic = {}

    for doc in docs:
        tokens.append(doc.text)
        pos_tags.append(doc.pos_)
        lemmas_dic.setdefault(doc.text, doc.lemma_)
    return {"tokens": tokens, "pos_tags": pos_tags, "lemmas": lemmas_dic}

def batch_tree2dic_optim(dec_tree, lemmas):
    rows = dec_tree.splitlines()[:-1] # skip the empty last index
    sentence = get_sentence(rows)
    #lemmas = get_lemmas_by_spacy(sentence)
    fields_mappings = []
    
    for row in rows:
        fields = row.split('\t')
        if fields[1].startswith("'") or fields[1].startswith("’"):
            lemma = fields[1][1:].lower()
        else:
            lemma = lemmas[fields[1]]
        dic = {
            'id': int(fields[0]),
            'text': fields[1],
            #'lemma': lemmas[fields[1]],
            'lemma': lemma,
            'upos': fields[3],
            'head': int(fields[6]),
            'deprel': fields[7]
        }
        fields_mappings.append(dic)
    return fields_mappings