import os
from pymongo import MongoClient
from image_match.web import templates, static


MONGO_CLIENT = MongoClient()
DB = MONGO_CLIENT['images']
DEFAULT_COLLECTION = 'eyeem_market'


USER_AGENT = 'ascribe image crawl'
CONNECT_TIMEOUT = 4
REQUEST_TIMEOUT = 4


TEMPLATE_PATH = os.path.dirname(templates.__file__)
STATIC_PATH = os.path.dirname(static.__file__)
FAVICON_ICO = '/'.join([STATIC_PATH, 'favicon.ico'])

BASE_URL = ''

COLLECTION_MAP = {
    'eyeem': 'eyeem_market',
    'direct2artist': 'direct2artist',
    'eyeem_global': 'crawl_images'
}

try:
    from image_match.web.local_settings import *
except ImportError:
    pass
