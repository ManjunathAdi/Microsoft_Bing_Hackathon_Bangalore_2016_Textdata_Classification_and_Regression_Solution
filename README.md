# Microsoft_Bing_Hackathon_Bangalore_2016_Textdata_Classification_and_Regression_Solution



My Solution and code details-

I have used neural word embeddings, paragraph vector and some sort of feature engineering for dense representation of every example/document for classification and regression.

4 scripts and details-

1.] Hack_W2vec.py file runs word2vec(skip-gram model) algorithm and gives word representation for all words including train and test data's summary information.

2.] DBOW.py file runs doc2vec(Dbow model) algorithm to get document level representation for all documnet's summary of both train and test data.

3.] Submission_Classification.py runs classification algorithm to categorize documents to their classes by taking feature represenation of documents obtained by concatenation of 3 things- 

a.] document representation of summary of document by averaging all wordvectors of the words of that document's summary,

b.] document represenation of summary of documents by Dbow doc2vec algorithm and

c.] A One-hot encoding feature matrix obtained by taking Author names of documents

Classifier used is Support vector machines.

Note: I didn't take title of documents into consideration because including it didn't improve accuracy

4.] Submission_Reg.py runs regression to predict the year of publication of the documents by taking feature representation of documents summary obtained by document representation of summary of documents by Dbow model of doc2vec algorithm.
Regressor used is a linear regression.


how to use/run?
First run Hack_W2vec.py and DBOW.py to get w2vec model and d2vec models.

Then these two models will be used in classification(script3) and regression(script4).
