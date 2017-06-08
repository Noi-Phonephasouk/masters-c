#Classification using SK learning and NKTL python libraries
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
from sklearn.cross_validation import KFold, train_test_split, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score, average_precision_score
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import nltk
import re
import sqlite3
import csv
import warnings
import datetime

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
                wrd_lst[i] = full_words[k]         
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
	stopwords = stopwords.words('english')

	lemmatizer = nltk.WordNetLemmatizer()
	stemmer = nltk.stem.porter.PorterStemmer()
	txt_sent = contractions_replace(txt_sent)
	txt_sent = re.sub(r"^\d+\W\s" , "", txt_sent)
	txt_sent = remove_puncz(txt_sent)
	txt_sent = txt_sent.lower()	
	new_sent_ls = word_tokenize(txt_sent)
	new_sent_ls = replace_abbr_word(new_sent_ls)
	 
	if len(new_sent_ls)<3:		
		return None
	else:
		# using word stem
		#words_lst = [stemmer.stem(w) for w in new_sent_ls if not w in stopwords]
		#new_sent = " ".join(words_lst)
		
		# use word lemma
		words_lst2 = [lemmatizer.lemmatize(w) for w in new_sent_ls if not w in stopwords]
		#words_lst2 = [w for w in new_sent_ls if not w in stopwords]
		
		new_sent = " ".join(words_lst2)		
		return new_sent

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

cur = curs.execute('''SELECT Sent_categorize.sentid, sent_group, truecatid FROM Sent_categorize, Sent_group 
			WHERE Sent_categorize.sentid = Sent_group.sentid ORDER BY Sent_categorize.sentid''')
rows = cur.fetchall()

ls_sentid = []
ls_review_sent = []
lst_sent_cluster = []
set_sent_lst = []
lst_data = []
lst_sent_classid = []
ls_sentid_1 = []
ls_review_sent_1 = []
lst_sent_feature_1 = []

for row in rows:	
	if row[0] not in ls_sentid:
		ls_sentid.append(int(row[0]))
		ls_review_sent.append({'sentid':int(row[0]), 'review_sent':row[1]})
	set_sent_lst.append({'sentid':int(row[0]), 'review_sent':row[1], 'classid':row[2]})
	
conn.commit()   

for j in range(0,len(ls_review_sent)):
	for i in range(0, len(set_sent_lst)):
		if ls_review_sent[j]['sentid'] == set_sent_lst[i]['sentid']:
			lst_sent_classid.append(set_sent_lst[i]['classid'])
	lst_data.append({'sentid': int(ls_review_sent[j]['sentid']), 'review_sent': feature_cull(ls_review_sent[j]['review_sent']), 'classid': lst_sent_classid})
	lst_sent_classid = []

data = DataFrame(lst_data, index = ls_sentid)
data_1 = DataFrame(lst_data[:650], index = ls_sentid[:650])
data_2 = DataFrame(lst_data[650:], index = ls_sentid[650:])

#randomize sample data
data = data.reindex(np.random.permutation(data.index))

#10 classes from 0-9

warnings.filterwarnings("ignore", category = DeprecationWarning)

pipeline = Pipeline([
	('vectorizer', CountVectorizer(ngram_range=(1,1))), # ngram_range=(1,4)
	('tfidf_transformer',  TfidfTransformer()),
	('classifier', OneVsRestClassifier(LinearSVC(C=1000000.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=1)))
	])
	
#Classifier: 
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

scores = []
precisions = []
recalls = []
#accuracies = []
#avg_pres = []
#avg_recs = []
class_report = []

k = 10 

k_fold = KFold(n = len(data), n_folds = k)
Y = MultiLabelBinarizer().fit_transform(data['classid'].values)
pipeline.fit(data['review_sent'].values, Y)
curs.execute('''UPDATE Sent_categorize SET predict_classid = Null ''')

dtname = datetime.datetime.now()
results = "result " +  str(dtname) + ".txt"
results = re.sub(r":" , ".", results)
print "writing...", results
file = open(results, "a")

file.write(str(pipeline.steps))
file.writelines("\n") 

file.write("KFold: k=" + str(k))
file.writelines("\n")   
nloop = 0

for train_indices, test_indices in k_fold:		
	train_text = data.iloc[train_indices]['review_sent'].values
	train_y = Y[train_indices]	
	test_text = data.iloc[test_indices]['review_sent'].values
	test_y = Y[test_indices]
	
	pipeline.fit(train_text, train_y)
	predictions = pipeline.predict(test_text)
	
	nloop += 1
	print "Loop: ", (nloop)
	for i, j in enumerate(test_indices):
		for k,l in enumerate(predictions[i]):
			if predictions[i][k] == 1:
				curs.execute('''UPDATE Sent_categorize SET predict_classid = ? WHERE sentid = ? ''', ( k, int(data.index[test_indices][i])))

	precision = precision_score(test_y, predictions)
	recall = recall_score(test_y, predictions)
	score = f1_score(test_y, predictions)
	
	print classification_report(test_y, predictions)
	scores.append(score)
	precisions.append(precision)
	recalls.append(recall)
		
	file.write("Loop: "+str(nloop))	
	file.writelines("\n")
	
	file.write("Classification Report ")
	file.writelines("\n")
	file.write(str(classification_report(test_y, predictions)))
	file.writelines("\n")

print('Total sentences classified:', len(data))
print('Precision:', sum(precisions)/len(precisions))
print('Recall:', sum(recalls)/len(recalls))
print('Score:', sum(scores)/len(scores))

conn.commit()

#write results to txt
file.writelines("\n\n")
file.write('Precision: %.3f%% ' % (sum(precisions)/len(precisions)*100.0))
file.writelines("\n")
file.write('Recall: %.3f%% ' % (sum(recalls)/len(recalls)*100.0))
file.writelines("\n")
file.write('FScore: %.3f%% ' % (sum(scores)/len(scores)*100.0))

# Testing on simulating data input
import csv
data_sim = open("categorize_simulation.csv", "r")    
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
print len(dataset_simulation)

input_simulation = MultiLabelBinarizer().fit_transform(dataset_simulation['classid'].values)
predictions_simulation = pipeline.predict(dataset_simulation['review_sent'].values)

precision_simulation = precision_score(input_simulation, predictions_simulation)
recall_simulation = recall_score(input_simulation, predictions_simulation)
F1_simulation = f1_score(input_simulation, predictions_simulation)

print "Classification Report "
print classification_report(input_simulation, predictions_simulation)

conn.close()
print "Task Done!"
