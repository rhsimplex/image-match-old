import os
from image_match.signature_database import SignatureCollection
from image_match.web import settings
from image_match.web.base import RequestHandler
from tempfile import NamedTemporaryFile
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
import tornado.web
import time



class Home(RequestHandler):

    def prepare(self):
        self.image_url = self.get_argument('image_url', None)
        self.collection = settings.DB[settings.DEFAULT_COLLECTION]

    @tornado.web.asynchronous
    def get(self):

        if self.image_url:
            http_client = AsyncHTTPClient()
            request = HTTPRequest(self.image_url,
                                  user_agent=settings.USER_AGENT,
                                  connect_timeout=settings.CONNECT_TIMEOUT,
                                  request_timeout=settings.REQUEST_TIMEOUT)
            http_client.fetch(request, self.handle_download)
        else:
            self.render('home.html', total='{}k'.format(self.collection.count() / 1000))

    def handle_download(self, response):
        if response.error:
            self.render('error.html', error=response.code)
        else:
            f = NamedTemporaryFile(delete=False)
            f.write(response.body)
            f.close()

            sc = SignatureCollection(self.collection, distance_cutoff=0.5)
            start_time = time.time()

            d = sc.similarity_search(f.name,
                                     process_timeout=1,
                                     maximum_matches_per_word=100)


            self.render('result.html',
                        result=d,
                        image_url=self.image_url,
                        request_time=response.request_time,
                        lookup_time=time.time() - start_time,
                        round=lambda x: round(x, 3))

            os.unlink(f.name)

