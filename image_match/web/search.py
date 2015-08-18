"""A unified view over the image/3d search API"""

import os.path
from image_match.web import settings
from image_match.signature_database import SignatureES


def _image_search(filename, origin):
    se = SignatureES(settings.ES)
    pass

def _three_d_search(filename, origin):
    pass


ext_to_func = {
    'stl': _three_d_search
}

def search(filename, origin=None):
    if origin == 'global':
        origin = None
    name, ext = os.path.splitext(filename)
    func = ext_to_func.get(ext, _image_search)
    return func(filename, origin)

