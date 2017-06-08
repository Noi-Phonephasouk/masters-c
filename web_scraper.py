from bs4 import BeautifulSoup
import cfscrape
import sqlite3
import re
import requests
from selenium import webdriver
from datetime import datetime
import time

browser = webdriver.Firefox()

#input the url (1)
browser.get("https://www.g2crowd.com/products/trello/reviews")
link = browser.find_elements_by_partial_link_text("Last")
for item in link:
	print item.get_attribute("href")
	total_pages = int(re.findall("\d+", item.get_attribute("href"))[1])
print "Retrieving reviews from ", total_pages, " pages..."

scraper = cfscrape.CloudflareScraper()

scontext = None

conn = sqlite3.connect('ureviewdata.sqlite')
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS Comments 
    (cid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, uid INTEGER NOT NULL, time TEXT, date TEXT, 
     title TEXT, review TEXT, UNIQUE (uid, date, title, review))''')

cur.execute('''CREATE TABLE IF NOT EXISTS Users 
    (uid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, wid INTEGER, username TEXT, rating TEXT, UNIQUE (wid, username, rating))''')

cur.execute('''CREATE TABLE IF NOT EXISTS Webs 
    (wid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, url TEXT UNIQUE)''')

urunning = 1

for i in range(total_pages):   
	#input the url (2)
	url = "https://www.g2crowd.com/products/trello/reviews?page={}".format(i+1)

	r = scraper.get(url).content

	data = r
	soup = BeautifulSoup(data, "lxml")

	cur.execute('''INSERT OR IGNORE INTO Webs (url) VALUES ( ? )''', ( url, ))
	cur.execute('SELECT wid FROM Webs WHERE url = ? ', (url, ))            
	web_id = cur.fetchone()[0] 
	
	for review in soup.find_all("div", itemprop= "review"):
		ureview_ls =[]
		ureview_ls_n = []
		
		for reviewtxt in review.find_all("div", itemprop= "reviewBody"):
			ureview_ls.extend(reviewtxt.stripped_strings)
			for item in ureview_ls:				
				if str(item[-1]) == "." :
					ureview_ls_n.append(item)
				elif str(item[-1]) == "?" :
					ureview_ls_n.append(item)
				elif str(item[-1]) == "!" :
					ureview_ls_n.append(item)
				else:
					ureview_ls_n.append(item + ".")			
			ureview = " ".join(ureview_ls_n)
			print str(ureview.encode('ascii', 'ignore'))
        
		for retitle in review.find_all("h3", attrs = {"class" : "review-list-heading"}):
			retitle = retitle.string
			if retitle==None:
				retitle = ""
			else:
				retitle = retitle.strip()
				retitle = str(retitle.encode('ascii', 'ignore'))	
			print retitle
        
		for time in review.find_all("time"):
			reviewdate = time.string        
			ireviewdate = datetime.strptime(reviewdate,'%B %d, %Y')
			reviewdate = datetime.strftime(ireviewdate, '%Y-%m-%d')      
            
		for item in review.find_all("div", attrs = re.compile(u'stars large stars-')):
			urate = item.attrs
			for item in urate:
				star = re.findall(r'\d+', urate[item][2])                
				print "rating out of 10: ", star[0]
				rerating = star[0]
              
		user=[]
        
		for username in review.find_all("img", alt = re.compile(u'review by')):
			user = username.get(u'alt')             
			euser = user.split('by ')
			if euser[1] == "Anonymous":
				reuser =  euser[1]+ str(urunning)
				urunning+=1
			else:
				reuser =  euser[1]
            
		cur.execute('''INSERT OR IGNORE INTO Users (wid, username, rating) 
            VALUES ( ?, ?, ? )''', ( web_id, (reuser), str(rerating), ) )
		cur.execute('SELECT uid FROM Users WHERE ((username = (?)) AND (wid = (?))) ', ((reuser), web_id ))
		user_id = cur.fetchone()[0]
        
		cur.execute('''INSERT OR IGNORE INTO Comments
            (uid, time, date, title, review) 
            VALUES ( ?, ?, ?, ?, ? )''', 
            ( user_id, str(reviewdate), str(reviewdate), str(retitle.encode('ascii', 'ignore')), str(ureview.encode('ascii', 'ignore')) ) )    
    
	conn.commit()
	scraper.close()
	
cur.close()
print "Task Done!!!"
