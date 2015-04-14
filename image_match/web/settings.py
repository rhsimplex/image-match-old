import os
from pymongo import MongoClient
from image_match.web import templates, static



MONGO_CLIENT = MongoClient()
DB = MONGO_CLIENT['images']
DEFAULT_COLLECTION = 'eyeem_market'

TEMPLATE_PATH = os.path.dirname(templates.__file__)
STATIC_PATH = os.path.dirname(static.__file__)
FAVICON_ICO = '/'.join([STATIC_PATH, 'favicon.ico'])

