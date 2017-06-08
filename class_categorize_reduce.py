#Classification using SK learning and NKTL libraries
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import nltk
import re
import csv
import warnings
import datetime

warnings.filterwarnings("ignore", category = DeprecationWarning)

def replace_abbr_word(wrd_lst):
    import csv
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
                wrd_lst[i]=full_words[k]         
    return wrd_lst

def contractions_replace(text):
	import pandas
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
	from nltk.corpus import stopwords
	stopwords = stopwords.words('english')+['general','nothing','able','way','mere','one','much','many','it', 'first', 'thing', 'things', 'something', 'everything', 'would', 'jira', 'trello', 'people','little','bit','yes',  'several','fact','thing', 'future','ocassion','mature','certain','lot','everyone','wish','non','atlassian','hard','enough','easy','old','folk','another','example','other','sad','eager','nightmare','anything','etc','anyone','uncommon','new','available','other','every','need','require','could','want','might','yet','also','dislike','like','sometimes','may','think','lack','cumbersome','annoy','better','wish']
	
	lemmatizer = nltk.WordNetLemmatizer()
	stemmer = nltk.stem.porter.PorterStemmer()
	txt_sent = contractions_replace(txt_sent)
	txt_sent = re.sub(r"^\d+\W\s" , "", txt_sent)
	txt_sent = remove_puncz(txt_sent)
	txt_sent = txt_sent.lower()
	
	new_sent_ls = word_tokenize(txt_sent)

	if len(new_sent_ls)<3:		
		return None
	else:
		# using word stem
		#words_lst = [stemmer.stem(w) for w in new_sent_ls if not w in stopwords]
		#new_sent = " ".join(words_lst)
		
		# using word lemma
		words_lst2 = [lemmatizer.lemmatize(w) for w in new_sent_ls if not w in stopwords]
		#words_lst2 = [w for w in new_sent_ls if not w in stopwords]
		
		new_sent = " ".join(words_lst2)
		
		return new_sent

import csv
data_merge = open("reduce_class_categorize.csv", "r")    
fh = csv.reader(data_merge)
sim_index = []
sim_data = []
    
ls_sentid = []
ls_review_sent = []
set_sent_lst = []
lst_sent_classid = []
lst_data = []

for line in fh:	
	if line[0] not in ls_sentid:
		ls_sentid.append(int(line[0]))
		ls_review_sent.append({'sentid':int(line[0]), 'review_sent':line[3]})
	set_sent_lst.append({'sentid':int(line[0]), 'review_sent':line[3], 'classid':line[1]})
	
for j in range(0,len(ls_review_sent)):
	for i in range(0, len(set_sent_lst)):
		if ls_review_sent[j]['sentid'] == set_sent_lst[i]['sentid']:
			lst_sent_classid.append(set_sent_lst[i]['classid'])
	lst_data.append({'sentid': int(ls_review_sent[j]['sentid']), 'review_sent': feature_cull(ls_review_sent[j]['review_sent']), 'classid': lst_sent_classid})
	lst_sent_classid = []
			
			
data = DataFrame(lst_data)
#randomize sample data
data = data.reindex(np.random.permutation(data.index))

#10 classes from 0-9

pipeline = Pipeline([
	('vectorizer', CountVectorizer(ngram_range=(1,2))), # ngram_range=(1,4)
	('tfidf_transformer',  TfidfTransformer()),
	('classifier', OneVsRestClassifier(LinearSVC(C=1000000.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=1)))
	])
	
Y = MultiLabelBinarizer().fit_transform(data['classid'].values)
pipeline.fit(data['review_sent'].values, Y)

#cross validation by train_test_split 
#define Testing proportion of sample
testsze = 0.20
print "Using", testsze*100.0,"% data for testing...\n"						

train_text, test_text, train_y, test_y, idx_train, idx_test = train_test_split(data['review_sent'].values, Y, data.index, test_size = testsze, random_state = 2)

pipeline.fit(train_text, train_y)
predictions = pipeline.predict(test_text)

train_test_precision= precision_score(test_y, predictions)
train_test_recall = recall_score(test_y, predictions)
train_test_F1 = f1_score(test_y, predictions)

print 
print "Classification Report "
print classification_report(test_y, predictions)

print "train_test_precision:", train_test_precision
print "train_test_recall:", train_test_recall
print "train_test_F1:", train_test_F1


#record result to a txt file
dtname = datetime.datetime.now()

#file name
results = "result " +  str(dtname) + ".txt"
results = re.sub(r":" , ".", results)

print "writing...", results

#classification parameters
file = open(results, "a")
file.write(str(pipeline.steps))
file.writelines("\n") 
file.write("Testing percentage: "+ str(testsze*100.0))
file.writelines("\n")   

#classification results
file.write("Classification Report")
file.writelines("\n")
file.write(classification_report(test_y, predictions))


# Testing on simulating data input selected from other sources of similar app types
import csv
data_sim = open("reduce_class_simulation.csv", "r")    
fh = csv.reader(data_sim)
sim_index = []
sim_data = []
    
ls_sentid = []
ls_review_sent = []
set_sent_lst = []
lst_sent_classid = []
lst_data = []

for line in fh:	
	if line[0] not in ls_sentid:
		ls_sentid.append(int(line[0]))
		ls_review_sent.append({'sentid':int(line[0]), 'review_sent':line[1]})
	set_sent_lst.append({'sentid':int(line[0]), 'review_sent':line[1], 'classid':line[4]})
	
for j in range(0,len(ls_review_sent)):
	for i in range(0, len(set_sent_lst)):
		if ls_review_sent[j]['sentid'] == set_sent_lst[i]['sentid']:
			lst_sent_classid.append(set_sent_lst[i]['classid'])
	lst_data.append({'sentid': int(ls_review_sent[j]['sentid']), 'review_sent': feature_cull(ls_review_sent[j]['review_sent']), 'classid': lst_sent_classid})
	lst_sent_classid = []	

dataset_simulation = DataFrame(lst_data, index=ls_sentid)

input_simulation = MultiLabelBinarizer().fit_transform(dataset_simulation['classid'].values)
predictions_simulation = pipeline.predict(dataset_simulation['review_sent'].values)

precision_simulation = precision_score(input_simulation, predictions_simulation)
recall_simulation = recall_score(input_simulation, predictions_simulation)
F1_simulation = f1_score(input_simulation, predictions_simulation)

print "Classification Report "
print classification_report(input_simulation, predictions_simulation)
