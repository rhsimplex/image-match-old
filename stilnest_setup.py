from image_match.signature_database import SignatureCollection
import sys
from pymongo import MongoClient
from os import walk
from os.path import join

"""
Script to setup a database from Stilnest directory structure
"""
def build_db(argv):
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

    sd = SignatureCollection(c)

    sd.add_images(paths)

if __name__ == "__main__":
    build_db(sys.argv)