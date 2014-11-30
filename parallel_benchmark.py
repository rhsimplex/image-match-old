from source.signature_database import SignatureCollection
from pymongo import MongoClient
from random import randint
from os import listdir
from os.path import join
from numpy.random import choice
import time
"""
Benchmarking script for parallel performance.

n_trials -- number of runs with each configuration
processes -- how many processes to spawn at a time

DB -- database name
COLLECTION -- collection name
DIR -- relative path to directory containing images
OUTPUT_FILE -- store results

output format is: number_of_processes iteration_of_first_result time_first_result time_exhaustive_search
"""


n_trials = 10
processes = [1,2,4,8,16,30]

DB = 'image_match'
COLLECTION = 'oxford_signatures'
DIR = 'oxbuild_images'
OUTPUT_FILE = 'results.csv'

client = MongoClient()
db = client[DB]
c = db[COLLECTION]

test_images = choice(listdir(DIR), size=n_trials*len(processes)).tolist()

sc = SignatureCollection(c)

with open(OUTPUT_FILE, 'w') as f:
    for n in processes:
        n_iterations = sc.N/n
        for trial in range(n_trials):
            s = sc.parallel_find(join(DIR,test_images.pop()), n_parallel_words=n)
            starttime = time.time()
            firstresult_time = None
            firstresult_iteration = None
            for iteration in range(n_iterations):
                result = s.next()
                if len(result) > 0 and firstresult_time is None:
                    firstresult_time = time.time()
                    firstresult_iteration = iteration
            allresults = time.time()
            f.write( '%i %i %f %f\n' % (n, firstresult_iteration, firstresult_time - starttime, allresults - starttime) )
        
