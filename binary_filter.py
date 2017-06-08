#Classification using SK learning and NKTL python libraries
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, train_test_split, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import csv
import re
import sqlite3
import datetime
import warnings

def replace_abbr_word(wrd_lst):
    #import csv
    abbr_file = open("abbreviations_list.csv", "r")    
    handler = csv.reader(abbr_file)
    abbr_words = []
    full_words = []
    
    for line in handler:
        if line[0] not in abbr_words:
            abbr_words.append(line[0]) 
        if line[1] not in full_words:
            full_words.append(line[1]) 
            
    abbr_file.close() 
    for i, j in enumerate(wrd_lst):
        for k,l in enumerate(abbr_words):
            if j.lower() == l:                
                wrd_lst[i] = full_words[k]         
    return wrd_lst

def contractions_replace(text):
	#import pandas
	text = text.lower()
	cont_file = open("contractions_list.csv", "r")    
	handler = csv.reader(cont_file)
	contraction_lst = []
		
	for line in handler:
		contraction_lst.append({'cont_wrd':line[0], 'full_wrd':line[1]})         
	cont_file.close()  
	
	cont_dataframe = DataFrame(contraction_lst)
	
	for i in range(0, len(cont_dataframe)):
		if cont_dataframe.iloc[i]['cont_wrd'] in text:
			text = text.replace(cont_dataframe.iloc[i]['cont_wrd'], cont_dataframe.iloc[i]['full_wrd'])
	return text
	
def remove_puncz(sentence):	 
	tokenizer = RegexpTokenizer(r"\w+")
	tokens = tokenizer.tokenize(sentence)
	return " ".join(tokens)

def feature_cull(txt_sent):
	lemmatizer = nltk.WordNetLemmatizer()
	stemmer = nltk.stem.porter.PorterStemmer()
	txt_sent = contractions_replace(txt_sent)
	txt_sent = re.sub(r"^\d+\W\s" , "", txt_sent)
	txt_sent = remove_puncz(txt_sent)
	txt_sent = txt_sent.lower()	
	new_sent_ls = word_tokenize(txt_sent)
	#new_sent_ls = replace_abbr_word(new_sent_ls)
	 
	if len(new_sent_ls)<3:		
		return None
	else:
		# use word stem
		words_lst = [stemmer.stem(w) for w in new_sent_ls]
		new_sent = " ".join(words_lst)
		
		# use word lemma
		#words_lst2 = [lemmatizer.lemmatize(w) for w in new_sent_ls]
		#new_sent = " ".join(words_lst2)
		
		return new_sent

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

#load truth set for evaluation
cur0 = curs.execute('''SELECT sentid, review_sent, truebiclassid FROM Sent_breakdown WHERE truebiclassid = 0 ORDER BY sentid''')
rows0 = cur0.fetchall()

ls_sentid_0 = []
ls_review_sent_0 = []
lst_sent_feature_0 = []

ls_sentid_1 = []
ls_review_sent_1 = []
lst_sent_feature_1 = []

for row in rows0:	
	if row[1] not in ls_review_sent_0:
		ls_sentid_0.append(row[0])
		ls_review_sent_0.append(row[1])
		lst_sent_feature_0.append(row[2])		
	conn.commit()   
	
cur1 = curs.execute('''SELECT sentid, review_sent, truebiclassid FROM Sent_breakdown WHERE truebiclassid = 1 ORDER BY sentid''')
rows1 = cur1.fetchall()

for row1 in rows1:
	if row1[1] not in ls_review_sent_1:
		ls_sentid_1.append(row1[0])	
		ls_review_sent_1.append(row1[1])	
		lst_sent_feature_1.append(row1[2])		
	conn.commit()  	

dataset_dict0 = []
index0 = []
dataset_dict1 = []
index1 = []
dataset_dict = []
index = []

for i, j in enumerate(ls_sentid_0):
	text_sent = str(feature_cull(ls_review_sent_0[i]))
	if text_sent == "None":
		continue
	else:
		sentiment_sent = vaderSentiment(text_sent)
		if sentiment_sent['compound']>0:
			sentiment_txt = " positive"
		elif sentiment_sent['compound']<0:
			sentiment_txt = " negative"
		else:
			sentiment_txt = " neutral"
			
		text_sent = text_sent  + "," + sentiment_txt			
		dataset_dict0.append({'sent_id':ls_sentid_0[i],'can_sent':text_sent, 'class':lst_sent_feature_0[i]})
		index0.append(ls_sentid_0[i])

for i,j in enumerate(ls_sentid_1):
	text_sent = str(feature_cull(ls_review_sent_1[i]))
	if text_sent == "None":
		continue
	else:
		sentiment_sent = vaderSentiment(ls_review_sent_1[i])
		if sentiment_sent['compound'] > 0:
			sentiment_txt = " positive"
		elif sentiment_sent['compound'] < 0:
			sentiment_txt = " negative"
		else:
			sentiment_txt = " neutral"
			
		text_sent = text_sent  + ", " + sentiment_txt				
		dataset_dict1.append({'sent_id':ls_sentid_1[i], 'can_sent':text_sent, 'class':lst_sent_feature_1[i]})
		index1.append(ls_sentid_1[i])
		
print "total non-featured sentences: ", len(dataset_dict0)
print "total featured sentences: ", len(dataset_dict1)

dataset_dict = dataset_dict0[:3778] + dataset_dict1[:700]
index = index0[:3778] + index1[:700]
#dataset_dict = dataset_dict0[:] + dataset_dict1[:]
#index = index0[:] + index1[:]
data = DataFrame(dataset_dict, index=index)

#randomize sample data
data = data.reindex(np.random.permutation(data.index))

#class 0 = 'Non Feature Request'
#class 1 = 'Feature Request'

pipeline = Pipeline([
	('vectorizer', CountVectorizer(ngram_range=(1,3))), # ngram_range=(1,4)
	#('vectorizer', HashingVectorizer(ngram_range=(1,4), non_negative=True)),
	('tfidf_transformer',  TfidfTransformer()),
	#('feature_selection', SelectKBest(chi2, k = 1)), #k =1,2,..
	#('feature_selection', SelectFromModel(LinearSVC())),
	#('scaling', StandardScaler()), ('pca', PCA(n_components=5)),
	#('svd', TruncatedSVD(algorithm='randomized',
     #   n_components=300, n_iter=5,
      #  random_state=2, tol=0.0)),
	('classifier', LinearSVC(C=1000000.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=1))
	])

#classifiers: 
	# SVC(C=1000000.0, gamma=0.0, kernel='rbf')
	# LinearSVC()
	# LinearSVC(C=1000000.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=1)
	# linear_model.SGDClassifier(loss='hinge',alpha=0.0001)
	# linear_model.LogisticRegression()
	# RandomForestClassifier(n_estimators=10)
	# MultinomialNB()
	# BernoulliNB()
	# KNeighborsClassifier(n_neighbors=3, weights="uniform")
	# tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)

print "Testing the classifer on sample dataset..."
	
pipeline.fit(data['can_sent'].values, data['class'].values)
warnings.filterwarnings("ignore", category = DeprecationWarning)

scores = []
precisions = []
recalls = []
accuracies = []
class_report = []

confusion = np.array([[0, 0],   
					[0, 0]])

testsze = 0.33

print "Using", testsze*100.0,"% data for testing...\n"						
# cross validation by train_test_split 
train_text, test_text, train_y, test_y, idx_train, idx_test = train_test_split(data['can_sent'].values, data['class'].values, data.index, test_size = testsze, random_state = 2)

pipeline.fit(train_text, train_y)
predictions = pipeline.predict(test_text)

train_test_precision = precision_score(test_y, predictions)
train_test_recall = recall_score(test_y, predictions)
train_test_F1 = f1_score(test_y, predictions)
#accuracy = accuracy_score(test_y, predictions)

print "Confusion Matrix "
print confusion_matrix(test_y, predictions)
print 
print "Classification Report "
print classification_report(test_y, predictions)

print "...recording data..."
#clear existing classification
curs.execute('''UPDATE Sent_breakdown SET predict_biclassid = Null ''')
#record new data
for i, j in enumerate(idx_test):		
	curs.execute('''UPDATE Sent_breakdown SET predict_biclassid = ? WHERE sentid = ? ''', (  int(predictions[i]), int(idx_test[i]), ) )
							
# test on Trello sample set
print "Applying the classifier on another sample dataset..."
datatest_newset = []
index_newset = []

datatest_newset = dataset_dict0[3778:] + dataset_dict1[700:]
index_newset = index0[3778:] + index1[700:]

data_new = DataFrame(datatest_newset, index = index_newset)

predictions_n = pipeline.predict(data_new['can_sent'].values)

precision_n= precision_score(data_new['class'].values, predictions_n)
recall_n = recall_score(data_new['class'].values, predictions_n)
F1_n = f1_score(data_new['class'].values, predictions_n)
#accuracy_n = accuracy_score(data_new['class'].values, predictions_n)

print "Confusion Matrix "
print confusion_matrix(data_new['class'].values, predictions_n)
print 
print "Classification Report "
print classification_report(data_new['class'].values, predictions_n)

#record the predict set
print "...recording data..."
for i, j in enumerate(index_newset):
	curs.execute('''UPDATE Sent_breakdown SET predict_biclassid = ? WHERE sentid = ? ''', (  int(predictions_n[i]), int(index_newset[i]), ) )
	
conn.commit() 

#write evaluation performance
dtname = datetime.datetime.now()
results = "result " +  str(dtname) + ".txt"
results = re.sub(r":" , ".", results)
print "writing...", results
file = open(results, "a")

file.write(str(pipeline.steps))
file.writelines("\n") 

file.write("Testing percentage: "+ str(testsze*100.0))
file.writelines("\n")   

file.write("Confusion Matrix")
file.writelines("\n")
file.write(str(confusion_matrix(test_y, predictions)))
file.writelines("\n")
file.write("Classification Report")
file.writelines("\n")
file.write(classification_report(test_y, predictions))
		
file.write("Apply the classifier to new dataset")
file.writelines("\n")

file.write("Confusion Matrix")
file.writelines("\n")
file.write(str(confusion_matrix(data_new['class'].values, predictions_n)))
file.writelines("\n")
file.write("Classification Report")
file.writelines("\n")
file.write(classification_report(data_new['class'].values, predictions_n))

file.close()

# Testing on simulating data input
import csv
data_sim = open("binary_filter_simulation.csv", "r")    
fh = csv.reader(data_sim)
sim_index = []
sim_data = []
sim_sent = ""
    
for line in fh:
	if line[0] not in sim_index:
		sim_index.append(int(line[0])) 
		sim_sent = str(feature_cull(line[1]))
		senti_sent = vaderSentiment(line[1])
		if senti_sent['compound']>0:
			senti_txt = " positive"
		elif senti_sent['compound']<0:
			senti_txt = " negative"
		else:
			senti_txt = " neutral"
			
		sim_sent = sim_sent  + ", " + senti_txt	
		
		sim_data.append({'can_sent':sim_sent, 'class':int(line[2])})
		
dataset_simulation = DataFrame(sim_data, index=sim_index)

predictions_simulation = pipeline.predict(dataset_simulation['can_sent'].values)

precision_simulation = precision_score(dataset_simulation['class'].values, predictions_simulation)
recall_simulation = recall_score(dataset_simulation['class'].values, predictions_simulation)
F1_simulation = f1_score(dataset_simulation['class'].values, predictions_simulation)


print "Confusion Matrix "
print confusion_matrix(dataset_simulation['class'].values, predictions_simulation)
print 
print "Classification Report "
print classification_report(dataset_simulation['class'].values, predictions_simulation)


conn.close()
print "Task Done!"