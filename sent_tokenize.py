import sqlite3
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.tokenize.punkt import PunktParameters
import nltk
import csv
import re

def contractions_replace(text):
    text=text.lower()
    cont_file = open("contractions_list.csv", "r")    
    handler = csv.reader(cont_file)
    cont_words = []
    full_words = []
    
    for line in handler:
        if line[0] not in cont_words:
            cont_words.append(line[0]) 
        if line[1] not in full_words:
            full_words.append(line[1]) 
            
    cont_file.close()  
  
    for item in cont_words:
        if item in text:
            tindex = cont_words.index(item)
            find_str = cont_words[tindex]
            
            replace_str = str(full_words[tindex])
            text = text.replace(find_str, replace_str)
    return text
def remove_puncz(sentence):	 
	tokenizer= RegexpTokenizer(r"\w+")
	tokens = tokenizer.tokenize(sentence)
	return " ".join(tokens)

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

curs.execute('''CREATE TABLE IF NOT EXISTS Sent_breakdown 
    (sentid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, cid INTEGER NOT NULL, 
	review_sent TEXT NOT NULL, dup_status INTEGER, truebiclassid INTEGER,
	predict_biclassid INTEGER, UNIQUE (cid, review_sent) )''')

cur = curs.execute('''SELECT review, cid FROM Comments ORDER BY cid''')
rows = cur.fetchall()

count =0	
countsent = 0
punkt_param = PunktParameters()
abbr = ['e.g', 'i.e','1','2','3','4','5','6','7', 'inc', 'smth']
punkt_param.abbrev_types = set(abbr)
ureview_txt_ls=[]

tokenizer=nltk.tokenize.punkt.PunktSentenceTokenizer(punkt_param)

for row in rows:
	ureview_txt = row[0]
	ureview_txt = re.sub(r"(etc.(?=\W+[a-z]))",r"etc",ureview_txt)	
			
	all_sentence_lst = tokenizer.tokenize(ureview_txt) # sentence tokenize
	for sent in all_sentence_lst:  
		sentn = re.sub(r"^\d+\W\s" , "", sent)
		sentn = remove_puncz(sentn)
		sentn = sentn.lower()
		sentn = contractions_replace(sentn)
		
		ureview_txt_ls = word_tokenize(sentn)
	
		if len(ureview_txt_ls)>2:	
			curs.execute('''INSERT OR IGNORE INTO Sent_breakdown (cid, review_sent) VALUES ( ?, ? )''', ( row[1], sent, ))
			print sent, ".... Extracted."
			countsent+=1
	conn.commit()    
	count+=1
	
conn.close()

print "Total tokenized comments:", count
print "Total tokenized sentences:", countsent

print "Task Done!"
