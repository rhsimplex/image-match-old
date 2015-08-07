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


@lru_cache_function(max_size=1024, expiration=60 * 60)
def get_count(index):
    return scale(settings.ES.count(index)['count'])


@lru_cache_function(max_size=1024, expiration=60 * 60)
def get_samples(index):
    if index != 'eyeem_market':
        return []

    samples = settings.ES.mget(index='eyeem_market',
                               body={'ids': [12, 3785, 3786, 4879, 13083,
                                             226786, 272669, 426077, 560792]},
                               fields=['path', '_id'])['docs']
    samples = [{'path': s['fields']['path'][0]} for s in samples]
    return samples


class Home(SimilaritySearchHandler):

    def handle_empty_query(self, index):
        samples = get_samples(index)
        self.normalize_results(samples)
        self.render('home.html',
                    market=self.market,
                    total='{}'.format(get_count(index)),
                    samples=samples)

    def handle_error(self, error):
        self.render('error.html', error=error, market=self.market)

    def handle_response(self, result, request_time, lookup_time):
        self.render('result.html',
                    result=result,
                    market=self.market,
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
