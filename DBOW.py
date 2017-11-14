#!/usr/bin/env python


# *************************************** #
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
import nltk.data
from nltk.corpus import stopwords
import logging
import gensim
from gensim.models import doc2vec

global i
i=1

class NatLangProcessing(object):
    """NatLangProcessing is class for processing raw text into segments for further learning"""
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    
    def Review_to_Words( self, review, remove_stopwords=False ):
        #
        review_text = re.sub("[^0-9]"," ", review)
        #  Convert words to lower case and split them
        words = review_text.lower().split()
        #  Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        # Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    def Review_to_Sentence( self, review, remove_stopwords=False ):
        # 
        global i
        # Remove HTML
        review = BeautifulSoup(review).get_text()
        #  
        review = re.sub("[^0-9]"," ", review)
        raw_sentences = self.tokenizer.tokenize(review.strip())
        #
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                string = "DOC_" + str(i)
                sentence = self.Review_to_Words( raw_sentence )
                senten = doc2vec.LabeledSentence(sentence,[string])
                sentences.append(senten)
                i +=1
        #
        return sentences
           
def execute(num_features=16, min_word_count=2, context=10):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("Loading Data\n")
    names=['RecordID', 'Topic', 'Pub_Year', 'Authors', 'Title', 'Summary']
    train = pd.read_csv("Desktop/scripts/BingHackMicroSoft/train.txt", names=names, header=None, delimiter="\t")
    
    truetest = pd.read_csv("Desktop/scripts/BingHackMicroSoft/test.txt", names=names, header=None, delimiter="\t")
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    NatLangProcessor = NatLangProcessing(tokenizer)
    sent=[]
    
    logging.info("Doing NLP on Data\n")
    for review in train["Summary"]:
        sent += NatLangProcessor.Review_to_Sentence(review)    
    for review in truetest["Summary"]:
        sent += NatLangProcessor.Review_to_Sentence(review)
    for review in truetest["Title"]:
        sent += NatLangProcessor.Review_to_Sentence(review)
    for review in train["Title"]:
        sent += NatLangProcessor.Review_to_Sentence(review)    
        
    logging.info("Training D2vec(DBOW) Algorithm:\n")
    num_workers = 4       # Number of threads to run in parallel
    downsampling = 1e-3   # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    model_dbow = gensim.models.Doc2Vec(min_count = min_word_count, window = context, size=num_features,  \
                 sample = downsampling, negative=5 , workers = num_workers, dm = 0)
    
    model_dbow.build_vocab(sent)

    for epoch in range(4):
        model_dbow.train(sent)
        model_dbow.alpha = model_dbow.alpha
        model_dbow.min_alpha = model_dbow.alpha
        
    model_dbow.save('DBOW_Model_Hack_16dim_2mc_4epochs')
   
execute(num_features=16, min_word_count=2, context=10)


