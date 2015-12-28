import csv
import re
import numpy as np

f = open("data/tweets.csv", "r")
reader = csv.reader(f, delimiter=',', quotechar='"')
headers = reader.next()

output = open("tweets.txt", "w")
for row in reader:
    if row[8]:  # retweeted_status_timestamp
        continue
    text = row[5]
    filtered = re.sub(r"@.+?([\t ]|$)", "", text)  # remove all @usernames
    filtered = re.sub(r"http.+[\t ]?", "", filtered)  # remove URLs
    filtered = filtered.strip()
    if len(filtered) > 0:
        output.write(filtered + "\n")

output.close()
f.close()

with open('tweets.txt','r') as source:
    data = source.readlines()
np.random.shuffle(data)
with open('tweets.txt','w') as target:
    target.writelines(data)