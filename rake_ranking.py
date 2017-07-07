from rake_nltk import Rake
import csv
import re
import operator
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import pandas as pd
import sqlite3

def replace_abbr_word(wrd_lst):
	'''To replace a list of abbreviated words to full words'''
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
	'''To replace a list of contracted words to full words'''
	text=text.lower()
	cont_file = open("contractions_list.csv", "r")    
	handler = csv.reader(cont_file)
	contraction_lst = []
		
	for line in handler:
		contraction_lst.append({'cont_wrd':line[0], 'full_wrd':line[1]})       
		
	cont_file.close()  	
	cont_dataframe = pd.DataFrame(contraction_lst)
	
	for i in range(0, len(cont_dataframe)):
		if cont_dataframe.iloc[i]['cont_wrd'] in text:
			text = text.replace(cont_dataframe.iloc[i]['cont_wrd'], cont_dataframe.iloc[i]['full_wrd'])   
	return text
	
def remove_puncz(sentence):
	'''To remove punctuations from a sentnece'''
	tokenizer= RegexpTokenizer(r"\w+")
	tokens = tokenizer.tokenize(sentence)
	return " ".join(tokens)
	
def feature_cull(txt_sent):
	'''To select key features by pre-processing techniques'''
	lemmatizer = nltk.WordNetLemmatizer()
	stemmer = nltk.stem.porter.PorterStemmer()
	txt_sent = contractions_replace(txt_sent)
	txt_sent= re.sub(r"^\d+\W\s" , "", txt_sent)
	txt_sent = remove_puncz(txt_sent)
	txt_sent = txt_sent.lower()	
	new_sent_ls = word_tokenize(txt_sent)
	new_sent_ls = replace_abbr_word(new_sent_ls)
	 
	if len(new_sent_ls)<3:		
		return None
	else:
		# use word stem
		#words_lst = [stemmer.stem(w) for w in new_sent_ls]
		#new_sent = " ".join(words_lst)
		
		# use word lemma
		#words_lst2 = [lemmatizer.lemmatize(w) for w in new_sent_ls]
		
		words_lst2 = [w for w in new_sent_ls]
		new_sent = " ".join(list(set(words_lst2)))
		
		return new_sent

def is_number(s):
	try:
		float(s) if '.' in s else int(s)
		return True
	except ValueError:
		return False

def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words
def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words

def build_stop_word_regex(stop_word_file_path):
	stop_word_list = load_stop_words(stop_word_file_path)
	stop_word_regex_list = []
	for word in stop_word_list:
		word_regex = r'\b' + word + r'(?![\w-])'
		stop_word_regex_list.append(word_regex)
	stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
	return stop_word_pattern

def generate_candidate_keywords(sentence_list, stopword_pattern):
	phrase_list = []
	phrase_sent = []
	itm = ''
	for s in sentence_list:
		tmp = re.sub(stopword_pattern, '|', s.strip())
		#print tmp, len(tmp)
		phrases = tmp.split("|")
		for phrase in phrases:
			phrase = phrase.strip().lower()
			if phrase != "" and phrase not in phrase_list:
				phrase_list.append(phrase)
		#print phrase_list
		for val in phrase_list:
			itm += val + " "
		
		phrase_sent.append(itm)
		itm = ''
		phrase_list = []
	return phrase_sent

def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/freq(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
    return word_score

def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

cur = curs.execute('''SELECT date, review, sent_group, sent_group.sentid,truecatid, truebiclassid, category, username,appname FROM sent_categorize,category,comments,sent_breakdown, sent_group, users, appname,webs WHERE sent_breakdown.cid = comments.cid AND sent_group.sentid=sent_breakdown.sentid AND sent_group.sentid=sent_categorize.sentid AND catid=truecatid AND users.uid=comments.uid AND users.wid=webs.wid AND webs.appid=appname.appid AND date BETWEEN "2010-01-01" AND "2016-12-31" AND appname ="Jira" ''')

#BETWEEN "2015-01-01" AND "2016-12-31" (YYY-MM-DD)- This allows using different date range to filter data

rows = cur.fetchall()

stoppath = build_stop_word_regex("SmartStoplist.txt")

sim_data = []
sim_data_cat = []
sim_sentls_set = []
lst_sent_classid = []

sim_sent_chk = []
sim_sentls_class =[]
lst_sent_idx = []

for row in rows:
	if not any(d.get('clsid', None) == int(row[4]) for d in sim_data_cat):
		sim_data_cat.append({"clsid": int(row[4]), "clsname":row[6]})
	if row[2] not in sim_sent_chk:
		sim_sent_chk.append(row[2])
		sim_sentls_set.append({'sentid':int(row[3]), 'review_sent':row[2], 'classid':int(row[4])})
		
conn.commit()
sim_data_cat.sort()

for j in range(0,len(sim_data_cat)):
	for i in range(0, len(sim_sentls_set)):
		if sim_data_cat[j]['clsid'] ==sim_sentls_set[i]['classid']:
			lst_sent_classid.append({'sent':str(sim_sentls_set[i]['review_sent']), 'sentid': int(sim_sentls_set[i]['sentid']), 'class':sim_sentls_set[i]['classid']})
			lst_sent_idx.append(int(sim_sentls_set[i]['sentid']))
			
	df = pd.DataFrame(lst_sent_classid, index = lst_sent_idx)	
	
	phraseList = generate_candidate_keywords(df.sent,stoppath)
	word_score = calculate_word_scores(phraseList)
	keywordcandidates = generate_candidate_keyword_scores(phraseList,word_score)
	sortedKeywords = sorted(keywordcandidates.iteritems(), key=operator.itemgetter(1), reverse=True)
	
	pl = pd.Series(phraseList, index=df.index)	
	df = df.assign(pl = pl.values)
	
	print "CATEGORY", sim_data_cat[j]['clsid'], sim_data_cat[j]['clsname'], "\n"
	
	for keyword in sortedKeywords[0:5]:
		for i in range(0, len(df)):		
			if keyword[0] == df.iloc[i]['pl']:
				print "Sentence" , str(sortedKeywords.index(keyword)+1)+ ": " + df.iloc[i]['sent'], "Keyword scores: %.3f" % keyword[1] #  + keyword[0]
	
	df.drop(df.index, inplace=True)
	pl.drop(pl.index, inplace=True)
	
	lst_sent_classid=[]
	lst_sent_idx=[]
	print "\n"
	

phraseList = generate_candidate_keywords(sim_sent_chk,stoppath)
word_score = calculate_word_scores(phraseList)
keywordcandidates = generate_candidate_keyword_scores(phraseList,word_score)

sortedKeywords = sorted(keywordcandidates.iteritems(), key=operator.itemgetter(1), reverse=True)
totalKeywords = len(sortedKeywords)

word_string = ""
for keyword in sortedKeywords[0:20]:
	word_string = word_string+keyword[0]

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(max_font_size=40, background_color='white').generate(word_string)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
conn.close()