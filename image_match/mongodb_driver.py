from signature_database_base import SignatureDatabaseBase
from signature_database_base import normalized_distance
from multiprocessing import cpu_count, Process, Queue
from multiprocessing.managers import Queue as managerQueue
import numpy as np


class SignatureMongo(SignatureDatabaseBase):
    def __init__(self, collection, *args, **kwargs):
        self.collection = collection
        # Extract index fields, if any exist yet
        if self.collection.count() > 0:
            self.index_names = [field for field in self.collection.find_one({}).keys()
                                if field.find('simple') > -1]

        super(SignatureMongo, self).__init__(*args, **kwargs)

    def search_single_record(self, rec, n_parallel_words=1, word_limit=None,
                             process_timeout=None, maximum_matches=1000):
        if n_parallel_words is None:
            n_parallel_words = cpu_count()

        if word_limit is None:
            word_limit = self.N

        initial_q = managerQueue.Queue()

        [initial_q.put({field_name: rec[field_name]}) for field_name in self.index_names[:word_limit]]

        # enqueue a sentinel value so we know we have reached the end of the queue
        initial_q.put('STOP')
        queue_empty = False

        # create an empty queue for results
        results_q = Queue()

        # create a set of unique results, using MongoDB _id field
        unique_results = set()

        l = list()

        while True:

            # build children processes, taking cursors from in_process queue first, then initial queue
            p = list()
            while len(p) < n_parallel_words:
                word_pair = initial_q.get()
                if word_pair == 'STOP':
                    # if we reach the sentinel value, set the flag and stop queuing processes
                    queue_empty = True
                    break
                if not initial_q.empty():
                    p.append(Process(target=get_next_match,
                                     args=(results_q,
                                           word_pair,
                                           self.collection,
                                           np.array(rec['signature']),
                                           self.distance_cutoff,
                                           maximum_matches)))

            if len(p) > 0:
                for process in p:
                    process.start()
            else:
                break

            # collect results, taking care not to return the same result twice

            num_processes = len(p)

            while num_processes:
                results = results_q.get()
                if results == 'STOP':
                    num_processes -= 1
                else:
                    for key in results.keys():
                        if key not in unique_results:
                            unique_results.add(key)
                            l.append(results[key])

            for process in p:
                process.join()

            # yield a set of results
            if queue_empty:
                break

        return l


    def insert_single_record(self, rec):
        self.collection.insert(rec)

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def index_collection(self):
        """Index a collection on words.

        """
        # Index on words
        self.index_names = [field for field in self.collection.find_one({}).keys()
                            if field.find('simple') > -1]
        for name in self.index_names:
            self.collection.create_index(name)


def get_next_match(result_q, word, collection, signature, cutoff=0.5, max_in_cursor=100):
    """Scans a cursor for word matches below a distance threshold.
    Exhausts a cursor, possibly enqueuing many matches
    Note that placing this function outside the SignatureCollection
    class breaks encapsulation.  This is done for compatibility with
    multiprocessing.
    Keyword arguments:
    result_q -- a multiprocessing queue in which to queue results
    word -- {word_name: word_value} dict to scan against
    collection -- a pymongo collection
    signature -- signature array to match against
    cutoff -- normalized distance limit (default 0.5)
    max_in_cursor -- if more than max_in_cursor matches are in the cursor,
        ignore this cursor; this column is not discriminatory
    """
    curs = collection.find(word, projection=['_id', 'signature', 'path'])

    # if the cursor has many matches, then it's probably not a huge help. Get the next one.
    if curs.count() > max_in_cursor:
        result_q.put('STOP')
        return

    matches = dict()
    while True:
        try:
            rec = curs.next()
            dist = normalized_distance(np.reshape(signature, (1, signature.size)), np.array(rec['signature']))[0]
            if dist < cutoff:
                matches[rec['_id']] = {'dist': dist, 'path': rec['path'], 'id': rec['_id']}
                result_q.put(matches)
        except StopIteration:
            # do nothing...the cursor is exhausted
            break
    result_q.put('STOP')
