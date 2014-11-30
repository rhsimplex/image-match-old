from source.signature_database import SignatureCollection
from pymongo import MongoClient
from random import randint
from os import listdir
from os.path import join
from numpy.random import choice
import time

n_trials = 10
processes = [1,2,4,8,16,30]

DB = 'image_match'
COLLECTION = 'oxford_signatures'
DIR = 'oxbuild_images'

client = MongoClient()
db = client[DB]
c = db[COLLECTION]

test_images = choice(listdir(DIR), size=n_trials*len(processes)).tolist()

sc = SignatureCollection(c)

for n in processes:
    n_iterations = sc.N/n
    for trial in range(n_trials):
        s = sc.parallel_find(join(DIR,test_images.pop()), n_parallel_words=n)
        starttime = time.time()
        firstresult = None
        for iteration in range(n_iterations):
            result = s.next()
            if len(result) > 0 and firstresult is None:
                firstresult = time.time()
        allresults = time.time()
        print '%i %f %f' % (n, firstresult - starttime, allresults - starttime) 
    
