import os
from image_match.signature_database import SignatureCollection
from image_match.web import settings
from image_match.web.base import RequestHandler
from tempfile import NamedTemporaryFile
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
import tornado.web
import time
import markdown



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
            samples = list(self.collection.find().limit(6))
            self.render('home.html', total='{}k'.format(self.collection.count() / 1000), samples=samples)

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



class Documentation(RequestHandler):
    def get(self, name):
        try:
            content = open(os.path.join(settings.TEMPLATE_PATH, 'md', name + '.md')).read()
        except IOError:
            raise tornado.web.HTTPError(404)

        self.render('documentation.html', raw_html=markdown.markdown(content))

