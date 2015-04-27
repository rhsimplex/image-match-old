from goldberg import ImageSignature
import numpy as np
from itertools import product
from multiprocessing import cpu_count
from elasticsearch import Elasticsearch
from operator import itemgetter
from datetime import datetime


class SignatureES(object):
    """Wrapper class for MongoDB collection.

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
        es.indices.create(index=index, ignore=400)

    def add_images(self, image_dir_or_list, drop_collection=False, limit=None, verbose=False,
                   insert_block_size=100000, n_processes=None):
        raise NotImplementedError

    def add_image(self, path, img=None, path_as_id=False):
        rec = make_record(path, self.gis, self.k, self.N, img,
                          integer_encoding=self.integer_encoding, path_as_id=path_as_id)
        rec['timestamp'] = datetime.now()
        self.es.index(index=self.index, doc_type=self.doc_type, body=rec)

    def parallel_find(self, path_or_signature, n_parallel_words=None, word_limit=None, verbose=False,
                      process_timeout=None, maximum_matches=1000):
        raise NotImplementedError

    def similarity_search(self, path, n_parallel_words=None, word_limit=None, all_results=True, all_orientations=False,
                          process_timeout=1, maximum_matches_per_word=1000):
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
        process_timeout -- how long to wait before joining a thread automatically, in seconds (default 1)
        maximum_matches -- ignore columns with maximum_matches or more (default 1000)
        """
        raise NotImplementedError
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
                                                                      word_limit=word_limit,
                                                                      process_timeout=process_timeout,
                                                                      maximum_matches=maximum_matches_per_word)))
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

