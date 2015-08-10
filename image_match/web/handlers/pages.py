import os
from image_match.web import settings
from image_match.web.base import RequestHandler, SimilaritySearchHandler
import tornado.web
import markdown
from lru import lru_cache_function


ORDERS = [(1e9, 'B'), (1e6, 'M'), (1e3, 'k')]


def scale(size):
    for n, l in ORDERS:
        if size > n:
            return '{}{}'.format(int(size / n), l)

def get_query(origin):
    if origin == 'global':
        filter_condition = {
                'missing': {
                    'field': 'origin'
                }
            }
    else:
        filter_condition = {
                'bool': {
                    'must': {
                        'term': {
                            'origin': origin
                            }
                        }
                    }
                }

    return {'query': {'filtered': {'filter': filter_condition}}}


@lru_cache_function(max_size=1024, expiration=60 * 60)
def get_count(origin):
    query = get_query(origin)
    return scale(settings.ES.count('images', body=query)['count'])


@lru_cache_function(max_size=1024, expiration=60 * 60)
def get_samples(origin):
    query = get_query(origin)
    samples = settings.ES.search(index='images',
                                 body=query,
                                 fields=['url', '_id'])['hits']['hits']
    samples = [{'url': s['fields']['url'][0]} for s in samples]
    return samples


class Home(SimilaritySearchHandler):

    def handle_empty_query(self, origin):
        samples = get_samples(origin)
        # self.normalize_results(samples)
        self.render('home.html',
                    market=self.origin,
                    total='{}'.format(get_count(origin)),
                    samples=samples)

    def handle_error(self, error):
        self.render('error.html', error=error, market=self.origin)

    def handle_response(self, result, request_time, lookup_time):
        self.render('result.html',
                    result=result,
                    market=self.origin,
                    image_url=self.image_url,
                    request_time=request_time,
                    lookup_time=lookup_time,
                    round=lambda x: round(x, 3))


class Documentation(RequestHandler):

    def get(self, market, name):
        path = os.path.join(settings.TEMPLATE_PATH, 'md', name + '.md')
        try:
            content = open(path).read()
        except IOError:
            raise tornado.web.HTTPError(404)
        template = tornado.web.template.Template(markdown.markdown(content))

        self.render('documentation.html',
                    raw_html=template.generate(market=market,
                                               BASE_URL=settings.BASE_URL),
                    market=market)
