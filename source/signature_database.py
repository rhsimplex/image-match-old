from goldberg import ImageSignature
import numpy as np
from pymongo.collection import Collection

class SignatureCollection(object):
    """Wrapper class for MongoDB collection.
    """
    def __init__(self, collection):
        if type(collection) is not Collection:
            raise TypeError('Expected MongoDB collection, got %r' % type(collection))
        self.collection = collection

    def add_images(self, image_dir):
        pass

    def add_image(self, path):
        pass
    
    @staticmethod
    def get_words(array, k, N):
        """Gets N words of length k from an array.

        Words may overlap.

        Keyword arguments:
        array -- array to split into words
        k -- word length
        N -- number of words
        """
        #generate starting positions of each word
        word_positions = np.linspace(0, array.shape[0],\
                N, endpoint=False).astype('int')

        #check that inputs make sense
        if k > array.shape[0]:
            raise ValueError('Word length cannot be longer than array length')
        if word_positions.shape[0] > array.shape[0]:
            raise ValueError('Number of words cannot be more than array length')

        #create empty words array
        words = np.zeros((N, k)).astype('int8')

        for i, pos in enumerate(word_positions):
            if pos + k <= array.shape[0]:
                words[i] = array[pos:pos+k]
            else:
                temp = array[pos:].copy()
                temp.resize(k)
                words[i] = temp

        return words

    @staticmethod
    def max_contrast(array):
        """Sets all positive values to one and all negative values to -1.

        Needed for first pass lookup on word table.

        Keyword arguments:
        array -- target array
        """
        array[array > 0] = 1
        array[array < 0] = -1

        return None
