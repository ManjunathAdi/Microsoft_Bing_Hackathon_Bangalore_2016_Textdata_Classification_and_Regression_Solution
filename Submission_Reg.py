#!/usr/bin/env python


# *************************************** #
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
import nltk.data
from nltk.corpus import stopwords
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn import linear_model
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures

class NatLangProcessing(object):
    """NatLangProcessing is class for processing raw text into segments for further learning"""
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    
    def Review_to_Words( self, review, remove_stopwords=False ):
        # 
        #  Remove HTML
        review_text = BeautifulSoup(review).get_text()
        # 
        review_text = re.sub("[^0-9]"," ", review_text)
        #  Convert words to lower case and split them
        words = review_text.lower().split()
        #  Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        # 
        return(words)

    # Define a function to split a review into parsed sentences
    def Review_to_Sentence( self, review, remove_stopwords=False ):
        #
        review = re.sub("[^0-9.]"," ", review)
        raw_sentences = self.tokenizer.tokenize(review.strip())
        #  Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( self.Review_to_Words( raw_sentence, ))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
        

class FeaureVecExtraction(object):
    def __init__(self,model,num_features):
        self.model = model
        self.num_features = num_features
    def makeFeatureVec(self, words):
        # Function to average all of the word vectors in a given paragraph
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((self.num_features,),dtype="float32")
        #
        nwords = 0.
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(self.model.index2word)
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,self.model[word])
        # Divide the result by the number of words to get the average
        featureVec = np.divide(featureVec,nwords)
        return featureVec
    def getAvgFeatureVecs(self, reviews):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        # Initialize a counter
        counter = 0.
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews),self.num_features),dtype="float32")
        # Loop through the reviews
        for review in reviews:
            # Print a status message every 1000th review
            if counter%100. == 0.:
                logging.info("Review %d of %d" % (counter, len(reviews)))
            # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = self.makeFeatureVec(review)
            # Increment the counter
            counter = counter + 1.
        return reviewFeatureVecs
        
def execute(num_features=32):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("Loading Data\n")
    names=['RecordID', 'Topic', 'Pub_Year', 'Authors', 'Title', 'Summary']
    Train = pd.read_csv("Desktop/scripts/BingHackMicroSoft/train.txt", names=names, header=None, delimiter="\t")
    
    Train.shape
    train=Train
    
    truetest = pd.read_csv("Desktop/scripts/BingHackMicroSoft/test.txt", names=names, header=None, delimiter="\t")

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    NatLangProcessor = NatLangProcessing(tokenizer)
    
    logging.info("Loading WOrd2Vec Model :\n")    
    # Initialize and train the model (this will take some time)
    model_summary = Word2Vec.load('Desktop/scripts/BingHackMicroSoft/Fulldata_W2vec_32dim_1mc')
    
    logging.info("Loading Doc2vec(DBOW) Model :\n")    
    # Initialize and train the model (this will take some time)
    dmodel_summary = Doc2Vec.load('Desktop/scripts/BingHackMicroSoft/DBOW_Model_Hack_16dim_2mc_4epochs')
    DtrainDataVecs = dmodel_summary.docvecs.doctag_syn0[0:4498,:]
    DtestDataVecs = dmodel_summary.docvecs.doctag_syn0[4498:5623,:]
    
    # ****** Create average vectors for the training and test sets
    FeaureVecExtractor = FeaureVecExtraction(model_summary, num_features)
    Summary_clean_train_reviews = []
    for review in train["Summary"]:
        Summary_clean_train_reviews.append( NatLangProcessor.Review_to_Words( review))
    logging.info("Creating average feature vecs for training reviews\n")
    trainDataVec = FeaureVecExtractor.getAvgFeatureVecs(Summary_clean_train_reviews)
    
    Summary_clean_test_reviews = []
    for review in truetest["Summary"]:
        Summary_clean_test_reviews.append( NatLangProcessor.Review_to_Words( review))
    logging.info("Creating average feature vecs for test reviews \n")
    testDataVec = FeaureVecExtractor.getAvgFeatureVecs(Summary_clean_test_reviews)
    
    #TrainDataVec = np.concatenate((trainDataVec, Train_matrix[0:4000,:]),axis=1)
    #TestDataVec = np.concatenate((testDataVec, Train_matrix[4000:4498,:]),axis=1)
    #trainDataVecs = np.concatenate((trainDataVec, DtrainDataVecs),axis=1)
    #testDataVecs = np.concatenate((testDataVec, DtestDataVecs),axis=1)
    trainDataVecs = DtrainDataVecs
    testDataVecs = DtestDataVecs
    print( trainDataVecs.shape)
    print( testDataVecs.shape)
    
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(trainDataVecs)
    predict_ = poly.fit_transform(testDataVecs)
    clf = linear_model.LinearRegression()
    clf.fit(X_, train["Pub_Year"])
    ans = clf.predict(predict_)
    
    #est = linear_model.LinearRegression()
    #est = linear_model.Ridge(alpha=0.1)
    
    #est = svm.SVR()
    #est = RandomForestRegressor(n_estimators=100, criterion='mse')
    #est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0, loss='ls')
    #est.fit(trainDataVecs, train["Pub_Year"])
    #ans = est.predict(testDataVecs)
    
    print( type(ans))
    ans = np.rint(ans)
#    MSE = mean_squared_error(test["Pub_Year"], ans) 
    #RMSE = mean_squared_error(test["Pub_Year"], ans)**0.5  
#    RMSE = sqrt(MSE)  
    output = pd.DataFrame(data={"record_id":truetest["RecordID"], "publication_year":ans})
    output.to_csv( "SubmissionReg_1.csv", index=False, sep='\t')

    # Test & extract results
#    logging.info("Results...\n")
#    logging.info("MSError = %f" %(MSE))
#    logging.info("RMSError = %f" %(RMSE))
    
execute(num_features=32)
"""

"""

