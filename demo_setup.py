from source.signature_database import SignatureCollection
from sys.stdout import flush
from os.path import split
from os import chdir, pardir, mkdir, remove
from urllib import urlretrieve
import tarfile

DB_NAME = 'image_match'
COLLECTION_NAME = 'oxford_signatures'
EXAMPLE_DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz'

try:
    import pymongo
except ImportError:
    print 'You need to install pymongo (http://api.mongodb.org/python/current/)'
    exit()

#-------------------CREATE COLLECTION--------------------
print 'Creating a collection...',
flush()
client = pymongo.MongoClient()
db = client[DB_NAME]
if COLLECTION_NAME not in db.collection_names():
    c = db[COLLECTION_NAME]
    print 'success!'
    print 'Collection %s created under database %s' \
            % (COLLECTION_NAME, DB_NAME)
else:
    print 'failed.'
    print 'A collection called %s already exists in database %s' \
            % (COLLECTION_NAME, DB_NAME)
    print 'Either remove the collection, or change the COLLECTION_NAME variable at the top of this script.'
    exit()

#-------------------DOWNLOAD DATA------------------------
archive_name = split(EXAMPLE_DATA_URL)[-1]
dir_name = archive_name[:archive_name.find('.')]
print 'Attempting to download %s...' % EXAMPLE_DATA_URL,
flush()
mkdir(dir_name)
chdir(dir_name)
try:
    urlretrieve(EXAMPLE_DATA_URL, archive_name)
except IOError:
    print 'The url %s may not be valid. Try another data set maybe?'
    exit()

print 'sucess!'
print 'Attempting to extract %s...' % archive_name,
flush()

tar = tarfile.open(archive_name)
tar.extractall()
print 'success! Deleting archive.'
os.remove(archive_name)
chdir(pardir())

#-------------------BUILD DATABASE-----------------------
print 'Adding images to database.'
sc = SignatureCollection(c)
sc.add_images(dir_name, verbose=True)
print 'Done!'
print 'Now run the included example ipython notebook for demonstration'
