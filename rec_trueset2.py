#write truth set into the database from a csv file
#multiclass
import sqlite3
import csv

datafile = open("truthset_categorize.csv", "r") 
file_handler = csv.reader(datafile)

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

curs.execute('''CREATE TABLE IF NOT EXISTS Sent_categorize 
    (sentid INTEGER, truecatid INTEGER, predict_classid INTEGER, UNIQUE (sentid, truecatid))''')

curs.execute('''DELETE FROM Sent_categorize''')
conn.commit()

for line in file_handler:	
	sent_id = line[0]		
	class_id = line[1]		
	curs.execute('''INSERT OR IGNORE INTO Sent_categorize (sentid, truecatid) VALUES ( ?, ? )''', ( sent_id, class_id, ))
conn.commit() 

conn.close()
datafile.close()

print "Task Done!"