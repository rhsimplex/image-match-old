import os
from image_match.web import templates, static
import elasticsearch


ES = elasticsearch.Elasticsearch(
    ['ec2-52-28-2-108.eu-central-1.compute.amazonaws.com',
     'ec2-52-28-126-90.eu-central-1.compute.amazonaws.com',
     'ec2-52-28-126-97.eu-central-1.compute.amazonaws.com'])

USER_AGENT = 'ascribe image crawl'
CONNECT_TIMEOUT = 4
REQUEST_TIMEOUT = 4

TEMPLATE_PATH = os.path.dirname(templates.__file__)
STATIC_PATH = os.path.dirname(static.__file__)
FAVICON_ICO = '/'.join([STATIC_PATH, 'favicon.ico'])

BASE_URL = ''

INDEX_MAP = {
    'photos': 'eyeem_market',
    'eyeem': 'eyeem_market',
    'direct2artist': 'direct2artist',
    'global': 'images'
}

try:
    from image_match.web.local_settings import *
except ImportError:
    pass
