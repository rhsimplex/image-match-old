from image_match.signature_database import SignatureCollection
import sys
import requests
from pymongo import MongoClient
from os import walk, spawnvp, P_WAIT, remove, listdir
from os.path import join, split, abspath
from time import time
from numpy import mean

# you will need blender 2.7x! see install instructions here:
# https://launchpad.net/~irie/+archive/ubuntu/blender
class StilnestCollection(SignatureCollection):
    """
    Stilnest specific version of SignatureCollection -- implements lookup function that takes an .stl URL

    e.g.

    import pymongo
    from image_match.stilnest_setup import StilnestCollection

    client = MongoClient()  # setup your MongoDB objects
    db = client['stilnest']
    c = db['signatures']

    stlc = StilnestCollection(c)  # instantiate this class

    example_stl_path = 'https://thingiverse-production.s3.amazonaws.com/assets/40/50/6e/c2/73/CoffeeSpoonHolder.stl'

    stlc.stilnest_lookup(example_stl_path)  # returns some stuff!

    """
    def stilnest_lookup(self, stl_file_URL, cutoff=0.1):
        #  make the cutoff something reasonable for this application
        self.distance_cutoff = cutoff

        # downlad and open file
        filename = split(stl_file_URL)[-1]
        with open(filename, 'w+') as f:
            download_start_time = time()
            stl_data = requests.get(stl_file_URL).content
            f.write(stl_data)

        # start blender script
        blender_start_time = time()
        L = ['blender', '-b', '-P', abspath('misc/stl2images.py'), '--',
             abspath(filename), abspath('.')]
        spawnvp(P_WAIT, 'blender', L)
        remove(filename)

        # start lookup
        database_start_time = time()
        views = {}

        # search each image against all orientations: 6 * 6 = 36 searches!
        for image_name in filter(lambda x: x.startswith(filename) and x.endswith('.png'),
                            listdir('.')):
            # search for any matches in this orientation
            views[image_name] = self.similarity_search(image_name,
                                                        n_parallel_words=10,
                                                        word_limit=10,
                                                        all_orientations=True)
            # remove temp file
            remove(image_name)

        # note the lookup completion time
        finish_time = time()

        # simplify result
        result = {}
        result['matches'] = []
        result['distances'] = []
        for key in views.keys():
            if views[key]['verdict'] is 'fail':
                result['matches'].append(split(split(views[key]['reason'][0]['path'])[0])[-1])
                result['distances'].append(views[key]['reason'][0]['dist'])

        # summarize matches
        result['match_summary'] = {}
        for i, match in enumerate(result['matches']):
            if match in result['match_summary'].keys():
                result['match_summary'][match].append(result['distances'][i])
            else:
                result['match_summary'][match] = []
                result['match_summary'][match].append(result['distances'][i])

        for key in result['match_summary'].keys():
            result['match_summary'][key] = mean(result['match_summary'][key])

        # add timing info
        result['timing'] = {'download': blender_start_time - download_start_time,
                          'blender': database_start_time - blender_start_time,
                          'lookup': finish_time - database_start_time}
        return result


"""
Script to setup a database from Stilnest directory structure
"""
def build_db(argv, n=9, cutoff=0.15, k=16, fix_ratio=True, crop_percentiles=(5, 95)):
    """
    Builds a signature collection from the stilnest directory structure

    Keyword arguments:
    argv -- commandline arguments. path to the top level stilnest dir
        all sub dirs will be checked for images
    """

    # get db and collection name if supplied
    if len(argv) > 2:
        db_name = argv[2]
        collection_name = argv[3]
    else:
        db_name = 'stilnest'
        collection_name = 'signatures'

    # set up a directory walk from the command line path
    t = walk(argv[1])

    # set up list to catch the paths
    paths = list()

    # catch all regular files (see os.walk)
    for d in t:
        if d[-1]:
            for x in d[-1]:
                paths.append(join(d[0], x))

    # access database
    client = MongoClient()
    db = client[db_name]
    c = db[collection_name]

    # drop collection
    c.drop()

    sd = SignatureCollection(c, distance_cutoff=cutoff, definite_match_cutoff=cutoff,
                             fix_ratio=fix_ratio, n_grid=n, k=k, crop_percentile=crop_percentiles)

    sd.add_images(paths)

if __name__ == "__main__":
    build_db(sys.argv)