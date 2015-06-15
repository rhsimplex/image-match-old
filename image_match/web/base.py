import tornado.web
import tornado.escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from image_match.web import settings
from image_match.signature_database import SignatureES
from tempfile import NamedTemporaryFile
import urllib
import os
import time
import markdown


def quote(uri):
    return urllib.quote(uri, safe='~@#$&*!+=:;,.?/\'')


class RequestHandler(tornado.web.RequestHandler):

    def get_template_namespace(self):
        ns = super(RequestHandler, self).get_template_namespace()
        ns.update({
            'BASE_URL': settings.BASE_URL,
            'escape': tornado.escape.url_escape,
            'quote': quote,
            'markdown': markdown.markdown
        })

        return ns


class SimilaritySearchHandler(RequestHandler):

    def prepare(self):
        self.image_url = self.get_argument('image_url', None)

    def normalize_results(self, d):
        for r in d:
            if 'path' not in r:
                r['path'] = r['_id']


    @tornado.web.asynchronous
    def get(self, market):
        if market.endswith('/'):
            market = market[:-1]

        self.market = market

        try:
            self.es_index = settings.INDEX_MAP[market]
        except KeyError:
            raise tornado.web.HTTPError(404)

        if self.image_url:
            http_client = AsyncHTTPClient()
            request = HTTPRequest(self.image_url,
                                  user_agent=settings.USER_AGENT,
                                  connect_timeout=settings.CONNECT_TIMEOUT,
                                  request_timeout=settings.REQUEST_TIMEOUT)
            http_client.fetch(request, self.handle_download)
        else:
            self.handle_empty_query(self.es_index)

    def handle_download(self, response):
        if response.error:
            self.handle_error(response.code)
        else:
            f = NamedTemporaryFile(delete=False)
            f.write(response.body)
            f.close()

            sc = SignatureES(settings.ES, index=self.es_index, distance_cutoff=0.5)
            start_time = time.time()

            # d = sc.similarity_search(f.name,
            #                          process_timeout=1,
            #                          maximum_matches_per_word=100)
            d = sc.bool_query(f.name, size=100)

            self.normalize_results(d)
            os.unlink(f.name)
            self.handle_response(d, response.request_time,
                                 time.time() - start_time)

    def handle_empty_query(self):
        raise tornado.web.HTTPError(404)

    def handle_error(self, error):
        raise NotImplementedError

    def handle_response(self, result, request_time, lookup_time):
        raise NotImplementedError
