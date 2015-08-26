import tornado.web
import tornado.escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from image_match.web import settings
from tempfile import NamedTemporaryFile
import urllib
import os
import time
import markdown
from image_match.web.search import search


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


class SearchHandler(RequestHandler):

    def prepare(self):
        self.url = self.get_argument('url', None)
        try:
            self.file = self.request.files['file'][0]
        except (IndexError, KeyError):
            self.file = None

        if self.url or self.file:
            _, self.ext = os.path.splitext(self.url or self.file['filename'])

    def _search(self, filename, origin=None, use_tineye=False):
        return self.do_search(filename, origin=None if origin == 'global' else origin, use_tineye=use_tineye)

    def do_search(self, filename, origin='global', use_tineye=False):
        raise NotImplementedError

    def get(self, origin):
        self.process(origin)

    def post(self, origin):
        self.process(origin)

    @tornado.web.asynchronous
    def process(self, origin):
        self.origin = origin

        if self.url:
            http_client = AsyncHTTPClient()
            request = HTTPRequest(self.url,
                                  user_agent=settings.USER_AGENT,
                                  connect_timeout=settings.CONNECT_TIMEOUT,
                                  request_timeout=settings.REQUEST_TIMEOUT)
            http_client.fetch(request, self.handle_download)
        elif self.file:
            self.handle_search(self.file['body'])
        else:
            self.handle_empty_query(self.origin)

    def handle_download(self, response):
        if response.error:
            self.handle_error(response.code)
        else:
            self.handle_search(response.body, response.request_time)

    def handle_search(self, file_body, request_time=0):
        f = NamedTemporaryFile(suffix=self.ext, delete=False)
        f.write(file_body)
        f.close()
        start_time = time.time()
        result = self._search(f.name, self.origin)
        os.unlink(f.name)
        self.handle_response(result, request_time, time.time() - start_time)

    def handle_empty_query(self, origin):
        raise tornado.web.HTTPError(404)

    def handle_error(self, error):
        raise NotImplementedError

    def handle_response(self, result, request_time, lookup_time):
        raise NotImplementedError


class SimilaritySearchHandler(SearchHandler):

    def do_search(self, filename, origin='global', use_tineye=False):
        return search(filename, origin, url=self.url, use_tineye=use_tineye)


class TineyeSearchHandler(SimilaritySearchHandler):
    def process(self, origin):
        self.origin = origin

        if self.url:
            start_time = time.time()
            result = self._search(self.url, self.origin, use_tineye=True)
            self.handle_response(result, 0, time.time() - start_time)
        else:
            self.handle_empty_query(self.origin)
