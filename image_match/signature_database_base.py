from image_match.goldberg import ImageSignature
from itertools import product
from operator import itemgetter
import numpy as np


class SignatureDatabaseBase(object):

    def search_single_record(self, rec):
        raise NotImplementedError

    def insert_single_record(self, rec):
        raise NotImplementedError

    def __init__(self, k=16, N=63, n_grid=9,
                 crop_percentile=(5, 95), distance_cutoff=0.45,
                 *signature_args, **signature_kwargs):

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

        self.distance_cutoff = distance_cutoff

        self.crop_percentile = crop_percentile

        self.gis = ImageSignature(n=n_grid, crop_percentiles=crop_percentile, *signature_args, **signature_kwargs)

    def add_image(self, path, img=None):
        rec = make_record(path, self.gis, self.k, self.N, img=img)
        self.insert_single_record(rec)

    def search_image(self, path, all_orientations=False, bytestream=False):
        img = self.gis.preprocess_image(path, bytestream)

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
            # compose all functions and apply on signature
            transformed_img = transform(img)

            # generate the signature
            transformed_record = make_record(transformed_img, self.gis, self.k, self.N)

            l = self.search_single_record(transformed_record)
            result.extend(l)
        r = sorted(np.unique(result).tolist(), key=itemgetter('dist'))
        s = set([x['id'] for x in r])
        for i, x in enumerate(r):
            if x['id'] not in s:
                r.pop(i)
            else:
                s.remove(x['id'])
        return r


def make_record(path, gis, k, N, img=None):
    """Makes a record suitable for database insertion.

    This non-class version of make_record is provided for
    CPU pooling. Functions passed to worker processes must
    be picklable.

    Keyword arguments:
    path -- path to image
    """
    record = dict()
    record['path'] = path
    if img is not None:
        signature = gis.generate_signature(img, bytestream=True)
    else:
        signature = gis.generate_signature(path)

    record['signature'] = signature.tolist()

    words = get_words(signature, k, N)
    max_contrast(words)

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


def normalized_distance(_target_array, _vec, nan_value=1.0):
    """Compute normalized distance to many points.

    Computes || vec - b || / ( ||vec|| + ||b||) for every b in target_array

    Keyword arguments:
    target_array -- N x m array
    vec -- array of size m
    nan_value -- value to replace 0.0/0.0 = nan with (default is 1.0, to take
                 those featureless images out of contention)
    """
    target_array = _target_array.astype(int)
    vec = _vec.astype(int)
    topvec = np.linalg.norm(vec - target_array, axis=1)
    norm1 = np.linalg.norm(vec, axis=0)
    norm2 = np.linalg.norm(target_array, axis=1)
    finvec = topvec / (norm1 + norm2)
    finvec[np.isnan(finvec)] = nan_value

    return finvec