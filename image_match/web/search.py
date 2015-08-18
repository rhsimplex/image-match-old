"""A unified view over the image/3d search API"""

import os
import os.path
import tempfile
from image_match.web import settings
from image_match.signature_database import SignatureES
from three_d_match import ThreeDSearch


def _image_search(filename, origin, url=None):
    engine = SignatureES(settings.ES)
    r = engine.bool_query(filename, origin=origin)
    return {
        'result': r,
        'url': url
    }


def _three_d_search(filename, origin, url=None):
    template = 'https://s3.eu-central-1.amazonaws.com/stilnest/human_views/{}.png'
    out_dir = tempfile.mkdtemp(dir=settings.TEMP_PATH)
    basename = os.path.basename(out_dir)
    engine = ThreeDSearch(settings.ES_ENDPOINTS, index_name='stilnest_3dmatch')
    engine.generate_images(filename, out_dir)
    matches = ThreeDSearch.best_single_image_list(engine.search_images(out_dir, origin))

    for match in matches:
        match['url'] = template.format(match['url'])

    result = {
        'urls': [os.path.join('/static', 'tmp', basename, img)
                 for img in os.listdir(out_dir)],
        'result': matches
    }
    return result


ext_to_func = {
    'stl': _three_d_search
}


def search(filename, origin=None, url=None):
    if origin == 'global':
        origin = None
    _, ext = os.path.splitext(filename)
    ext = ext[1:]
    func = ext_to_func.get(ext, _image_search)
    return func(filename, origin, url=url)
