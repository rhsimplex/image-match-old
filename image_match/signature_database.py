from goldberg import ImageSignature
import numpy as np
from itertools import product
from multiprocessing import cpu_count, Process, Queue
from multiprocessing.managers import Queue as managerQueue
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from operator import itemgetter
from datetime import datetime
from os import listdir
from os.path import join

class SignatureES(object):
    """Wrapper class for ElasticSearch object.

    See section 2 of Goldberg et al, available at:

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf
    """
    def __init__(self, es, index='images', doc_type='image', k=16, N=63, n_grid=9,
                 distance_cutoff=0.5, definite_match_cutoff=0.40,
                 integer_encoding=True, fix_ratio=False,
                 crop_percentile=(5, 95)):
        """Initialize SignatureCollection object

        Keyword arguments:
        es -- the ElasticSearch object
        index -- the name of the elastic search index (default 'images')
        doc_type -- the name of the elastic search doc_type (default 'image')
        k -- word length
        N -- number of words (default 63; max 64 indexes for MongoDB, need to leave
            one for _id_)
        n_grid -- size of n x n grid to compute signatures on (default 9)
        distance_cutoff -- maximum normalized distance between image signature and
            match signatures (default 0.5)
        definite_match_cutoff -- cutoff for definite match. Used in method
            similarity_search for match verdict(defualt 0.45)
        integer_encoding -- save words as integers instead of arrays (default True)
        fix_ratio -- enforce square grids. Not recommended except for square images
            ...use with care!
        """

        # Check that collection is a ElasticSearch object
        if type(es) is not Elasticsearch:
            raise TypeError('Expected ElasticSearch object, got %r' % type(es))

        self.es = es

        # Check integer inputs
        if type(k) is not int:
            raise TypeError('k should be an integer')
        if type(N) is not int:
            raise TypeError('N should be an integer')
        if type(n_grid) is not int:
            raise TypeError('n_grid should be an integer')

        self.k = k
        self.N = N
        self.n_grid = n_grid

        # Check float input
        if type(distance_cutoff) is not float:
            raise TypeError('distance_cutoff should be a float')
        if distance_cutoff < 0.:
            raise ValueError('distance_cutoff should be > 0 (got %r)' % distance_cutoff)
        if type(definite_match_cutoff) is not float:
            raise TypeError('definite_match_cutoff should be a float')
        if definite_match_cutoff > distance_cutoff:
            raise ValueError('definite_match_cutoff should be < %f (got %r)' % (distance_cutoff,
                                                                                definite_match_cutoff))

        self.distance_cutoff = distance_cutoff
        self.definite_match_cutoff = definite_match_cutoff

        # Check bool input
        if type(integer_encoding) is not bool:
            raise TypeError('integer_encoding should be boolean (got %r)')\
                  % type(integer_encoding)

        if type(fix_ratio) is not bool:
            raise TypeError('fix_ratio should be boolean')

        # need to add input checking
        self.crop_percentile = crop_percentile

        # Use default ImageSignature parameters for now, only allowing fix_ratio to vary
        self.gis = ImageSignature(n=n_grid, fix_ratio=fix_ratio, crop_percentiles=crop_percentile)

        self.integer_encoding = integer_encoding

        # Create ES index, if none exists
        self.index = index
        self.doc_type = doc_type
        es.indices.create(index=self.index, ignore=400)

        # Extract index fields, if any exist yet
        try:
            example_res = self.es.search(index=self.index, size=1)
            if example_res['hits']['total'] > 0:
                self.index_names = [field for field in
                                    example_res['hits']['hits'][0]['_source'].keys()
                                    if field.find('simple') > -1]

        except (NotFoundError, IndexError):
            # index doesn't exist yet or is empty
            pass

    def add_images(self, image_dir='.'):
        """Minimal batch adding, ignore non-images, all exceptions"""
        for path in listdir(image_dir):
            try:
                self.add_image(join(image_dir, path))
            except Exception as e:
                pass

    def add_image(self, path, img=None, path_as_id=False):
        rec = make_record(path, self.gis, self.k, self.N, img,
                          integer_encoding=self.integer_encoding, path_as_id=path_as_id)
        rec['timestamp'] = datetime.now()
        self.es.index(index=self.index, doc_type=self.doc_type, body=rec)

    def parallel_find(self, path_or_signature, n_parallel_words=None, word_limit=None, verbose=False,
                      process_timeout=None, maximum_matches=1000):
        """Makes an iterator to gets tne next match(es).

        Multiprocess find

        Keyword arguments:
        path_or_signature -- path to image or signature array
        n_parallel_words -- number of words to scan in parallel (default: number of physical processors times 2)
        word_limit -- limit the number of words to search (default None)
        process_timeout -- how long to wait before joining a thread automatically (default None)
        maximum_matches -- ignore columns with maximum_matches or more (default 1000)
        """
        if n_parallel_words is None:
            n_parallel_words = cpu_count()

        if word_limit is None:
            word_limit = self.N

        # check if an array (signature) was passed. If so, generate the words here:
        if type(path_or_signature) is np.ndarray:
            record = dict()
            words = get_words(path_or_signature, self.k, self.N)
            max_contrast(words)
            for i in range(self.N):
                record[''.join(['simple_word_', str(i)])] = words[i].tolist()
            record['signature'] = path_or_signature

        # otherwise, generate the record in the usual way
        else:
            # Don't encode words yet because we need to compute stds
            record = make_record(path_or_signature, self.gis, self.k, self.N, integer_encoding=False)

        # Generate standard deviations of each word vector.
        stds = dict()
        for word_name in self.index_names:
            stds[word_name] = np.std(record[word_name])

        keys = list(stds.keys())
        vals = list(stds.values())

        # Fill a queue with {word: word_val} pairs in order of std up to the word limit
        initial_q = managerQueue.Queue()
        while len(stds) > (self.N - word_limit):
            max_val = max(vals)
            max_pos = vals.index(max_val)
            max_word = keys[max_pos]

            stds.pop(max_word)
            vals.pop(max_pos)
            keys.pop(max_pos)

            if verbose:
                print '%s %f' % (max_word, max_val)
                print record[max_word]

            initial_q.put(
                {max_word: words_to_int(np.array([record[max_word]]))[0]}
            )

        # enqueue a sentinel value so we know we have reached the end of the queue
        initial_q.put('STOP')
        queue_empty = False

        if verbose:
            print 'Queue length: %i' % initial_q.qsize()

        # create an empty queue for results
        results_q = Queue()

        # create a set of unique results, using MongoDB _id field
        unique_results = set()

        # begin iterator
        while True:
            # if all there are no more cursors, kill iterator
            if queue_empty:
                # Queue.empty is not reliable. The iteration will be stopped by the sentinel value
                # (the last item in the queue)
                raise StopIteration

            # build children processes, taking cursors from in_process queue first, then initial queue
            p = list()
            while len(p) < n_parallel_words:
                word_pair = initial_q.get()
                if word_pair == 'STOP':
                    # if we reach the sentinel value, set the flag and stop queuing processes
                    queue_empty = True
                    break
                if not initial_q.empty():
                    p.append(Process(target=get_next_matches,
                                     args=(results_q,
                                           word_pair,
                                           self.es,
                                           self.index,
                                           record['signature'],
                                           self.distance_cutoff,
                                           maximum_matches)))

            if verbose:
                print '%i fresh cursors remain' % initial_q.qsize()

            if len(p) > 0:
                for process in p:
                    process.start()
            else:
                raise StopIteration

            # collect results, taking care not to return the same result twice
            l = list()
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
            yield l

    def similarity_search(self, path,
                          bytestream=False,
                          n_parallel_words=1,
                          word_limit=10,
                          all_results=True,
                          all_orientations=False,
                          process_timeout=1,
                          maximum_matches_per_word=100):
        """Performs similarity search on image

        Essentially a wrapper for parallel_find.

        path -- path or url to image
        n_parallel_words -- number of parallel processes to use (default CPU count)
        word_limit -- limit number of words to search against (default 2 * CPU count)
        result_limit
        all_orientations -- check image against all 90 degree rotations, mirror images, color inversions, and
            combinations thereof (default False)
        process_timeout -- how long to wait before joining a thread automatically, in seconds (default 1)
        maximum_matches -- ignore columns with maximum_matches or more (default 1000)
        """
        # get initial image
        img = self.gis.preprocess_image(path, bytestream=bytestream)

        if n_parallel_words is None:
            n_parallel_words = cpu_count()

        if word_limit is None:
            word_limit = 2 * cpu_count()
            if word_limit < self.N**0.5:
                word_limit = int(round(self.N**0.5))

        if all_orientations:
            # initialize an iterator of composed transformations
            inversions = [lambda x: x, lambda x: -x]

            mirrors = [lambda x: x, np.fliplr]

            # an ugly solution for function composition
            rotations = [lambda x: x,
                         np.rot90,
                         lambda x: np.rot90(x, 2),
                         lambda x: np.rot90(x, 3)]

            # cartesian product of all possible orientations
            orientations = product(inversions, rotations, mirrors)

        else:
            # otherwise just use the identity transformation
            orientations = [lambda x: x]

        # try for every possible combination of transformations; if all_orientations=False,
        # this will only take one iteration
        result = []

        orientations = np.unique(np.ravel(list(orientations)))
        for transform in orientations:
            # compose all functions and apply on signature, in a woefully inelegant way
            transformed_img = transform(img)

            # generate the signature
            transformed_signature = self.gis.generate_signature(transformed_img)

            l = reduce(lambda a, b: a + b, list(self.parallel_find(transformed_signature,
                                                  n_parallel_words=n_parallel_words,
                                                  word_limit=word_limit,
                                                  process_timeout=process_timeout,
                                                  maximum_matches=maximum_matches_per_word)))
            l = sorted(l, key=itemgetter('dist'))
            result.extend(l)
        return np.unique(result).tolist()


def make_record(path, gis, k, N, img=None, integer_encoding=True, path_as_id=False):
    """Makes a record suitable for database insertion.

    This non-class version of make_record is provided for 
    CPU pooling. Functions passed to worker processes must
    be picklable.

    Keyword arguments:
    path -- path to image
    """
    record = dict()
    if path_as_id:
        record['_id'] = path
    else:
        record['path'] = path
    if img is not None:
        signature = gis.generate_signature(img)
    else:
        signature = gis.generate_signature(path)

    record['signature'] = signature.tolist()

    words = get_words(signature, k, N)
    max_contrast(words)

    if integer_encoding:
        words = words_to_int(words)

    for i in range(N):
        record[''.join(['simple_word_', str(i)])] = words[i].tolist()

    return record


def get_words(array, k, N):
    """Gets N words of length k from an array.

    Words may overlap.

    Keyword arguments:
    array -- array to split into words
    k -- word length
    N -- number of words
    """
    # generate starting positions of each word
    word_positions = np.linspace(0, array.shape[0],
                                 N, endpoint=False).astype('int')

    # check that inputs make sense
    if k > array.shape[0]:
        raise ValueError('Word length cannot be longer than array length')
    if word_positions.shape[0] > array.shape[0]:
        raise ValueError('Number of words cannot be more than array length')

    # create empty words array
    words = np.zeros((N, k)).astype('int8')

    for i, pos in enumerate(word_positions):
        if pos + k <= array.shape[0]:
            words[i] = array[pos:pos+k]
        else:
            temp = array[pos:].copy()
            temp.resize(k)
            words[i] = temp

    return words


def words_to_int(word_array):
    """Converts a simplified word to an integer

    Encodes a k-byte word to int (as those returned by max_contrast).
    First digit is least significant.
    
    Returns dot(word + 1, [1, 3, 9, 27 ...] ) for each word in word_array

    e.g.:
    [ -1, -1, -1] -> 0
    [ 0,   0,  0] -> 13
    [ 0,   1,  0] -> 16

    Keyword arguments:
    word_array -- N x k array of simple words
    """
    width = word_array.shape[1]

    # Three states (-1, 0, 1)
    coding_vector = 3**np.arange(width)

    # The 'plus one' here makes all digits positive, so that the
    # integer represntation is strictly non-negative and unique
    return np.dot(word_array + 1, coding_vector)


def max_contrast(array):
    """Sets all positive values to one and all negative values to -1.

    Needed for first pass lookup on word table.

    Keyword arguments:
    array -- target array
    """
    array[array > 0] = 1
    array[array < 0] = -1

    return None


def normalized_distance(target_array, vec):
    """Compute normalized distance to many points.

    Computes || vec - b || / ( ||vec|| + ||b||) for every b in target_array

    Keyword arguments:
    target_array -- N x m array
    vec -- array of size m
    """
    # use broadcasting
    return np.linalg.norm(vec - target_array, axis=1)\
        / (np.linalg.norm(vec, axis=0) + np.linalg.norm(target_array, axis=1))


def get_next_matches(result_q, word, es, index_name, signature, cutoff=0.5, max_in_cursor=100, timeout=None):
    """Scans an index for word matches below a distance threshold.

    Note that placing this function outside the SignatureCollection
    class breaks encapsulation.  This is done for compatibility with
    multiprocessing.

    Keyword arguments:
    result_q -- a multiprocessing queue in which to queue results
    word -- {word_name: word_value} dict to scan against
    es -- an elasticsearch object
    index_name -- an elasticsearch index name
    signature -- signature array to match against
    cutoff -- normalized distance limit (default 0.5)
    max_in_cursor -- if more than max_in_cursor matches are in the cursor,
        ignore this cursor; this column is not discriminatory
    """
    res = es.search(index=index_name,
                    body={
                        'query': {
                            'match': word
                        }
                    },
                    fields=['path', 'signature'],
                    size=max_in_cursor)

    # if the cursor has many matches or is empty, then it's probably not a huge help. Get the next one.
    if res['hits']['total'] > max_in_cursor or res['hits']['total'] == 0:
        result_q.put('STOP')
        return

    # make a signature array, n x len(sig)
    signature_target_array = np.array([item['fields']['signature'] for item in res['hits']['hits']], dtype='int8')

    # make the target array, len(sig)
    signature_vec = np.array(signature, dtype='int8')

    # compute the distances
    distances = normalized_distance(signature_target_array, signature_vec).tolist()

    matches = dict()
    for i, dist in enumerate(distances):
        if dist < cutoff:
            id_str = res['hits']['hits'][i]['_id']
            matches[id_str] = {'dist': dist,
                               'path': res['hits']['hits'][i]['fields']['path'],
                               'id': id_str}
            result_q.put(matches)
    result_q.put('STOP')