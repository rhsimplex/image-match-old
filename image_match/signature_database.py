from goldberg import ImageSignature
import numpy as np
import csv
from itertools import product
from multiprocessing import cpu_count, Pool
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from elasticsearch.exceptions import NotFoundError, ConnectionTimeout
from operator import itemgetter
from datetime import datetime
from os import listdir
from os.path import join
from functools import partial

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

    def add_images(self, image_dir='.', ids_file=None, bulk_num=500, n_processes=4, offset=0, ignore_all=True):
        """Minimal batch adding, ignore non-images, all exceptions

        @:param image_dir path to directory holding images
        @:param ids_file unique ids associated with file names with rows formatted [id], ["local_path"], ["url"]
        """
        if ids_file:
            partial_mr = partial(make_record, gis=self.gis, k=self.k, N=self.N)
            with open(ids_file, 'rb') as csvfile:
                recordreader = csv.reader(csvfile, quotechar='"')
                pool = Pool(n_processes)
                end_reached = False
                for i in range(offset):
                    recordreader.next()
                while not end_reached:
                    ids = []
                    urls = []
                    local_paths = []
                    for i in range(bulk_num):
                        try:
                            _id, local_path, url = recordreader.next()
                            ids.append(_id)
                            urls.append(url)
                            local_paths.append(local_path)
                        except StopIteration:
                            end_reached = True
                    try:
                        results = pool.map(partial_mr, local_paths)
                    except IOError:
                        # file is missing
                        pass
                    except Exception as e:
                        if ignore_all:
                            continue
                        else:
                            raise e

                    to_insert = []
                    timestamp = datetime.now()
                    for i, result in enumerate(results):
                        result['path'] = urls[i]
                        result['timestamp'] = timestamp
                        to_insert.append(
                            {
                                '_index': self.index,
                                '_type': self.doc_type,
                                '_id': ids[i],
                                '_source': result
                            })
                    while True:
                        try:
                            _, errs = bulk(self.es, to_insert)
                            break
                        except (ConnectionTimeout, BulkIndexError):
                            continue

        else:
            for path in listdir(image_dir):
                try:
                    self.add_image(join(image_dir, path))
                except Exception as e:
                    pass

    def verify_database(self, ids_file, offset=0, ignore_timeout=True, verbose=False):
        """Verify database ids from list

        @:param ids_file unique ids associated with file names with rows formatted where at least the first column is id
        :return: none, lines are printed to stdout
        """
        line_no = 0
        with open(ids_file, 'rb') as csvfile:
            try:
                recordreader = csv.reader(csvfile, quotechar='"')
                if offset:
                    for i in range(offset):
                        recordreader.next()
                        line_no += 1
                for row in recordreader:
                    try:
                        res = self.es.search_exists(index=self.index, doc_type=self.doc_type,
                                              body={'query':
                                                        {'term':
                                                             {'_id': row[0]}
                                                        }
                                                   }
                                              )
                        if verbose:
                            print '{} {}'.format(row[0], res)
                    except NotFoundError:
                        print ', '.join(row)
                    finally:
                        line_no += 1
                        if verbose:
                            print line_no

            except Exception as e:
                raise RuntimeError('Fail at line {} caused by {}'.format(line_no, str(type(e))))

    def add_image(self, path, img=None, path_as_id=False):
        rec = make_record(path, self.gis, self.k, self.N, img,
                          integer_encoding=self.integer_encoding, path_as_id=path_as_id)
        rec['timestamp'] = datetime.now()
        self.es.index(index=self.index, doc_type=self.doc_type, body=rec)

    def parallel_find(self, path_or_signature, n_parallel_words=None, word_limit=None, verbose=False,
                      process_timeout=None, maximum_matches=100):
        """Makes an iterator to gets tne next match(es).

        Multiprocess find

        Keyword arguments:
        path_or_signature -- path to image or signature array
        n_parallel_words -- number of words to scan in parallel (default: number of physical processors times 2)
        word_limit -- limit the number of words to search (default None)
        process_timeout -- how long to wait before joining a thread automatically (default None)
        maximum_matches -- ignore columns with maximum_matches or more (default 100)
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
        initial_q = []
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

            initial_q.append(
                {max_word: words_to_int(np.array([record[max_word]]))[0]}
            )

        while True:
            body = [{'index': self.index, 'type': self.doc_type}]
            try:
                for i in range(n_parallel_words):
                    # noinspection PyTypeChecker
                    body.append({'query':
                                    {'term':
                                        initial_q.pop()
                                    },
                                    'size': maximum_matches,
                                    'fields': ['signature', 'path']
                                }
                                )
            except IndexError:
                raise StopIteration

            res = self.es.msearch(body=body)

            #  get number of results to allocate space for arrays with this beautifully opaque functional cudgel
            n_rows = sum(filter(lambda x: x < maximum_matches, map(lambda x: len(x['hits']['hits']), res['responses'])))
            signatures_array = np.zeros((n_rows, self.gis.sig_length), dtype=np.int8)
            paths = []
            ids = []

            #  build paths list and signature array
            index = 0
            for response in res['responses']:
                if len(response['hits']['hits']) < maximum_matches:
                    for hit in response['hits']['hits']:
                        signatures_array[index] = hit['fields']['signature']
                        paths.append(hit['fields']['path'][0])
                        ids.append(hit['_id'])
                        index += 1

            #  to avoid calculating redundant distances, eliminate duplicates based on path
            paths, ids = map(np.array, (paths, ids))
            _, unique_indices = np.unique(paths, return_index=True)
            paths, ids = map(lambda x: x[unique_indices], (paths, ids))

            #  compute the distances in one pass and filter bad matches
            dist = normalized_distance(signatures_array[unique_indices], record['signature'])
            final_mask = dist < self.distance_cutoff

            yield map(lambda x: dict(zip(['id', 'path', 'dist'], x)), zip(ids[final_mask],
                                                                          paths[final_mask],
                                                                          dist[final_mask]))

    def similarity_search(self, path,
                          bytestream=False,
                          n_parallel_words=1,
                          word_limit=10,
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