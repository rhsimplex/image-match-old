import os
from image_match.web import settings
from image_match.web.base import RequestHandler, SimilaritySearchHandler, TineyeSearchHandler
import tornado.web
import markdown
from lru import lru_cache_function


ORDERS = [(1e9, 'B'), (1e6, 'M'), (1e3, 'k')]


def scale(size):
    if not size:
        return None
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


class PageHandler(SimilaritySearchHandler):

    def handle_empty_query(self, origin):
        raise NotImplementedError

    def handle_error(self, error):
        self.render('error.html', error=error, market=self.origin)

    def handle_response(self, result, request_time, lookup_time):
        if 'error' in result:
            self.handle_error(result['error'])
            return
        self.render('result.html',
                    result=result,
                    market=self.origin,
                    request_time=request_time,
                    lookup_time=lookup_time,
                    round=lambda x: round(x, 3))


class Home(PageHandler):

    def handle_empty_query(self, origin):
        samples = get_samples(origin)
        self.render('home.html',
                    market=self.origin,
                    total=get_count(origin),
                    samples=samples)

class TineyeHome(TineyeSearchHandler):

    def handle_empty_query(self, origin):
        samples = get_samples(origin)
        self.render('home.html',
                    market=self.origin,
                    total=get_count(origin),
                    samples=samples)
        raise NotImplementedError

    def handle_error(self, error):
        self.render('error.html', error=error, market=self.origin)

    def handle_response(self, result, request_time, lookup_time):
        if 'error' in result:
            self.handle_error(result['error'])
            return
        self.render('result.html',
                    result=result,
                    market=self.origin,
                    request_time=request_time,
                    lookup_time=lookup_time,
                    round=lambda x: round(x, 3))

class StilnestHome(PageHandler):

    def handle_empty_query(self, origin):
        self.render('stilnest.html', market=self.origin)


class Documentation(RequestHandler):

    def get(self, market, name):
        if name == 'api' and market == 'all':
            name = 'tineye-api'
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
