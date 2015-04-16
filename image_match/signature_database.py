from goldberg import ImageSignature
import numpy as np
from itertools import product
from os import listdir
from os.path import join
from multiprocessing import Pool, cpu_count, Process, Queue
from multiprocessing.managers import Queue as managerQueue
from pymongo.collection import Collection
from functools import partial
from operator import itemgetter


class SignatureCollection(object):
    """Wrapper class for MongoDB collection.

    See section 2 of Goldberg et al, available at:

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf
    """
    def __init__(self, collection, k=16, N=63, n_grid=9,
                 distance_cutoff=0.5, definite_match_cutoff=0.40,
                 integer_encoding=True, fix_ratio=False,
                 crop_percentile=(5, 95)):
        """Initialize SignatureCollection object

        Keyword arguments:
        collection -- MongoDB collection. Can be empty or populated.
            Currently there is no schema checking, so passing a coll-
            ection not created by an instance of SignatureCollection
            may cause bizarre behavior
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

        # Check that collection is a MongoDB collection
        if type(collection) is not Collection:
            raise TypeError('Expected MongoDB collection, got %r' % type(collection))

        self.collection = collection

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

        # Extract index fields, if any exist yet
        if self.collection.count() > 0:
            self.index_names = [field for field in self.collection.find_one({}).keys()
                                if field.find('simple') > -1]

    def add_images(self, image_dir_or_list, drop_collection=False, limit=None, verbose=False,
                   insert_block_size=100000, n_processes=None):
        """Bulk adds images to database.

        Probably not very efficient, but fine for prototyping.

        Keyword arguments:
        image_dir -- directory with images (note: directory should contain only
            images; no checking of any kind is done)
        drop_collection -- remove current entries prior to insertion (default False)
        limit -- maximum records to create, to nearest block (default None)
        verbose -- enable console output (default False)
        insert_block_size -- number of records to bulk insert at a time (default 1000)
        n_processes -- number of threads to use. If None, use number of CPUs (default none)
            Note that this only applies to record generation.  Parallelism for insertion
            is handled by the MongoDB internals.
        """
        if drop_collection:
            self.collection.remove({})
            self.collection.drop_indexes()
            if verbose:
                print 'Collection contents dropped.'

        if n_processes is None:
            n_processes = cpu_count()

        if verbose:
            print 'Using %i processes.' % (2 * n_processes)

        pool = Pool(2 * n_processes)

        if type(image_dir_or_list) is str:
            image_dir = image_dir_or_list
            image_paths = map(lambda x: join(image_dir, x), listdir(image_dir))
        elif type(image_dir_or_list) is list:
            image_paths = image_dir_or_list
        else:
            raise ValueError('A directory name or list of files is required')

        if limit is None:
            limit = len(image_paths)
        if verbose:
            print '%i files found in %s.' % (limit, image_dir)

        partial_mr = partial(make_record,
                             gis=self.gis, k=self.k, N=self.N)
        # Insert image signatures and words
        for i in range(0, len(image_paths), insert_block_size):
            if i < limit:
                recs = pool.map(partial_mr, image_paths[i : i + insert_block_size])
            else:
                recs = pool.map(partial_mr, image_paths[i : limit])
            self.collection.insert(recs)

            if verbose:
                print 'Inserted %s records.' % str(i + insert_block_size)

        if verbose:
            print 'Total %i records inserted.' % self.collection.count()

        self.index_collection(verbose=verbose)

    def index_collection(self, verbose=False):
        """Index a collection on words.

        Keyword arguments:
        verbose -- enable console output (default False)
        """

        # Index on words
        self.index_names = [field for field in self.collection.find_one({}).keys()\
                if field.find('simple') > -1]
        for name in self.index_names:
            self.collection.create_index(name)
            if verbose:
                print 'Indexed %s' % name

    def add_image(self, path):
        """Inserts a single image.

        Creates indexes if this is the first entry.

        Keyword arguments:
        path -- path to image
        """
        self.collection.insert(make_record(path, gis=self.gis, k=self.k, N=self.N))

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def parallel_find(self, path_or_signature, n_parallel_words=None, word_limit=None, verbose=False):
        """Makes an iterator to gets tne next match(es).

        Multiprocess find

        Keyword arguments:
        path_or_signature -- path to image or signature array
        n_parallel_words -- number of words to scan in parallel (default: number of physical processors times 2)
        word_limit -- limit the number of words to search (default None)
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
                    p.append(Process(target=get_next_match,
                                     args=(results_q,
                                           word_pair,
                                           self.collection,
                                           record['signature'],
                                           self.distance_cutoff)))

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

            # yield a set of results
            yield l

    def similarity_search(self, path, n_parallel_words=None, word_limit=None, all_results=True, all_orientations=False):
        """Performs similarity search on image

        Essentially a wrapper for parallel_find.

        Returns a dict with the result:
        {"verdict": pass|fail|pending,
        "reason": dict of definite or possible matches}

        path -- path or url to image
        n_parallel_words -- number of parallel processes to use (default CPU count)
        word_limit -- limit number of words to search against (default 2 * CPU count)
        result_limit
        all_orientations -- check image against all 90 degree rotations, mirror images, color inversions, and
            combinations thereof (default False)
        """
        # get initial image
        img = self.gis.preprocess_image(path)

        if n_parallel_words is None:
            n_parallel_words = cpu_count()

        if word_limit is None:
            word_limit = 2 * cpu_count()
            if word_limit < self.N**0.5:
                word_limit = int(round(self.N**0.5))

        if all_results:
            l = reduce(lambda a, b: a + b, list(self.parallel_find(path,
                                                                      n_parallel_words=n_parallel_words,
                                                                      word_limit=word_limit)))
            l = sorted(l, key=itemgetter('dist'))
            return l

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
            orientations = [[lambda x: x]]

        # initialize a list to hold borderline cases
        borderline_cases = list()

        # try for every possible combination of transformations; if all_orientations=False,
        # this will only take one iteration
        for transforms in orientations:
            # compose all functions and apply on signature, in a woefully inelegant way
            transformed_img = img
            for transform in transforms:
                transformed_img = transform(transformed_img)

            # generate the signature
            transformed_signature = self.gis.generate_signature(transformed_img)

            # initialize the iterator
            s = self.parallel_find(transformed_signature, n_parallel_words=n_parallel_words, word_limit=word_limit)

            while True:
                try:
                    result = s.next()
                    # investigate if any results are returned
                    if result:
                        for entry in result:
                            # if the result is closer than the definite cutoff, return immediately
                            if entry['dist'] < self.definite_match_cutoff:
                                return {'verdict': 'fail', 'reason': [entry]}
                            elif entry['dist'] < self.distance_cutoff:
                                borderline_cases.append(entry)

                except StopIteration:
                    # iterator is exhausted, no matches found. Break out and try next orientation, if possible
                    break

        # return pass if borderline_cases is empty, otherwise pending
        if borderline_cases:
            return {'verdict': 'pending', 'reason': borderline_cases}
        else:
            return {'verdict': 'pass', 'reason': []}


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


def get_next_match(result_q, word, collection, signature, cutoff=0.5, max_in_cursor=1000):
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
    curs = collection.find(word, fields=['_id', 'signature', 'path'])

    # if the cursor has many matches, then it's probably not a huge help. Get the next one.
    if curs.count() > max_in_cursor:
        result_q.put('STOP')
        return

    matches = dict()
    while True:
        try:
            rec = curs.next()
            dist = normalized_distance([signature], np.array(rec['signature'], dtype='int8'))[0]
            if dist < cutoff:
                matches[rec['_id']] = {'dist': dist, 'path': rec['path'], 'id': rec['_id']}
                result_q.put(matches)
        except StopIteration:
            # do nothing...the cursor is exhausted
            break
    result_q.put('STOP')

