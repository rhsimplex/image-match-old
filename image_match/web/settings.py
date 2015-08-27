import os
from image_match.web import templates, static
import elasticsearch


ES_ENDPOINTS = ['ec2-52-28-2-108.eu-central-1.compute.amazonaws.com',
                'ec2-52-28-126-90.eu-central-1.compute.amazonaws.com',
                'ec2-52-28-126-97.eu-central-1.compute.amazonaws.com']

USER_AGENT = 'ascribe image crawl'
CONNECT_TIMEOUT = 4
REQUEST_TIMEOUT = 4

TEMPLATE_PATH = os.path.dirname(templates.__file__)
STATIC_PATH = os.path.dirname(static.__file__)
TEMP_PATH = os.path.join(os.path.dirname(static.__file__), 'tmp')
FAVICON_ICO = '/'.join([STATIC_PATH, 'favicon.ico'])

BASE_URL = ''

ORIGIN_MAP = {
    'photos': 'eyeem.com',
    'eyeem': 'eyeem.com',
    'direct2artist': 'direct2artist.com',
    'global': None
}

try:
    from image_match.web.local_settings import *  # NOQA
except ImportError:
    pass

ES = elasticsearch.Elasticsearch(ES_ENDPOINTS)
