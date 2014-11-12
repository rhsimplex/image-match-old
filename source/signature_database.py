from goldberg import ImageSignature
import numpy as np
from os import listdir
from os.path import join
from pymongo.collection import Collection

class SignatureCollection(object):
    """Wrapper class for MongoDB collection.

    See section 2 of Goldberg et al, available at:

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf
    """
    def __init__(self, collection, k=16, N=63):
        """Initialize SignatureCollection object

        Keyword arguments:
        collection -- MongoDB collection. Can be empty or populated.
            Currently there is no schema checking, so passing a coll-
            ection not created by an instance of SignatureCollection
            may cause bizarre behavior
        k -- word length
        N -- number of words (default 63; max 64 for MongoDB, need to leave
            one for _id_)
        """

        #Check that collection is a MongoDB collection
        if type(collection) is not Collection:
            raise TypeError('Expected MongoDB collection, got %r' % type(collection))
        
        self.collection = collection
        
        #Use default ImageSignature parameters for now
        self.gis = ImageSignature()

        #Check integer inputs
        if type(k) is not int:
            raise TypeError('k should be an integer')
        if type(N) is not int:
            raise TypeError('N should be an integer')

        self.k = k
        self.N = N


    def add_images(self, image_dir, drop_collection=False, limit=None, verbose=False,\
            insert_block_size=100):
        """Bulk adds images to database.

        Probably not very efficient, but fine for prototyping.

        Keyword arguments:
        image_dir -- directory with images (note: directory should contain only
            images; no checking of any kind is done)
        drop_collection -- remove current entries prior to insertion
        limit -- maximum records to create (not implemented)
        verbose -- enable console output
        insert_block_size -- number of records to bulk insert at a time
        """
        if drop_collection:
            self.collection.remove({})
            self.collection.drop_indexes()
            if verbose:
                print 'Collection contents dropped.'

        image_paths = map(lambda x: join(image_dir, x), listdir(image_dir))
        
        #Insert image signatures and words
        for i in range(0, len(image_paths), insert_block_size):
            self.collection.insert(\
                    map(self.make_record, image_paths[i : i + insert_block_size]))
            if verbose:
                print 'Inserted %i records.' % i
        
        if verbose:
            print 'Total %i records inserted.' % self.collection.count()

        #Index on words
        index_names = [field for field in self.collection.find_one({}).keys()\
                if field.find('simple') > -1]
        for name in index_names:
            self.collection.create_index(name)
            if verbose:
                print 'Indexed %s' % name

    def add_image(self, path):
        pass

    def make_record(self, path):
        """Makes a record suitable for database insertion.

        Keyword arguments:
        path -- path to image
        """
        record = {}
        record['path'] = path
        signature = self.gis.generate_signature(path)
        record['signature'] = signature.tolist()
        words = self.get_words(signature, self.k, self.N)
        self.max_contrast(words)
        
        for i in range(self.N):
            record[''.join(['simple_word_', str(i)])] = words[i].tolist()

        return record
        
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
