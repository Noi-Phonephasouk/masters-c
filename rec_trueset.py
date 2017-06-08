#write truth set into the database from a csv file
#binary
import sqlite3
import csv

datafile = open("truthset_filter.csv", "r") 
file_handler = csv.reader(datafile)

conn = sqlite3.connect('ureviewdata.sqlite')
curs = conn.cursor()

#drop any existing data of train/test sets
curs.execute('''UPDATE Sent_breakdown SET truebiclassid = Null ''')

for line in file_handler:	
	sent_idx = line[0]		
	class_t = line[2]
	curs.execute('''UPDATE Sent_breakdown SET truebiclassid = ? WHERE sentid = ? ''', ( class_t,sent_idx ) )
		
conn.commit() 

conn.close()
datafile.close()

print "Task Done!"