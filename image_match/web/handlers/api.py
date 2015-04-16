import os
import time
import tornado.web
from image_match.signature_database import SignatureCollection
from image_match.web import settings
from image_match.web.base import RequestHandler
from tempfile import NamedTemporaryFile
from tornado.httpclient import AsyncHTTPClient, HTTPRequest



class Search(RequestHandler):

    def prepare(self):
        self.collection = settings.DB[settings.DEFAULT_COLLECTION]

    @tornado.web.asynchronous
    def get(self, url):
        self.image_url = url
        http_client = AsyncHTTPClient()
        request = HTTPRequest(self.image_url,
                user_agent=settings.USER_AGENT,
                connect_timeout=settings.CONNECT_TIMEOUT,
                request_timeout=settings.REQUEST_TIMEOUT)
        http_client.fetch(request, self.handle_download)

    def handle_download(self, response):
        if response.error:
            self.write({'error': 'Timeout downloading the image'})
        else:
            f = NamedTemporaryFile(delete=False)
            f.write(response.body)
            f.close()

            sc = SignatureCollection(self.collection, distance_cutoff=0.5)

            d = sc.similarity_search(f.name,
                                     process_timeout=1,
                                     maximum_matches_per_word=100)


            os.unlink(f.name)

            self.write({'result': d})
            self.finish()
