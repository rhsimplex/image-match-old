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


SIGNATURE_MATCH = SignatureES(settings.ES)


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
        self.url = self.get_argument('url', None)

    @tornado.web.asynchronous
    def get(self, origin):
        if origin.endswith('/'):
            origin = origin[:-1]

        self.origin = origin

        # try:
        #     self.origin = settings.ORIGIN_MAP[origin]
        # except KeyError:
        #     raise tornado.web.HTTPError(404)

        if self.url:
            http_client = AsyncHTTPClient()
            request = HTTPRequest(self.url,
                                  user_agent=settings.USER_AGENT,
                                  connect_timeout=settings.CONNECT_TIMEOUT,
                                  request_timeout=settings.REQUEST_TIMEOUT)
            http_client.fetch(request, self.handle_download)
        else:
            self.handle_empty_query(self.origin)

    def handle_download(self, response):
        if response.error:
            self.handle_error(response.code)
        else:
            f = NamedTemporaryFile(delete=False)
            f.write(response.body)
            f.close()

            start_time = time.time()
            d = SIGNATURE_MATCH.bool_query(f.name, size=9, origin=None if self.origin == 'global' else self.origin)
            os.unlink(f.name)
            self.handle_response(d, response.request_time,
                                 time.time() - start_time)

    def handle_empty_query(self, origin):
        raise tornado.web.HTTPError(404)

    def handle_error(self, error):
        raise NotImplementedError

    def handle_response(self, result, request_time, lookup_time):
        raise NotImplementedError


class ThreeDimSimilaritySearchHandler(SimilaritySearchHandler):
    pass
