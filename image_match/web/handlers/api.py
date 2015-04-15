import tornado.web
from image_match.signature_database import SignatureCollection
from image_match.web import settings
from image_match.web.base import RequestHandler



class Search(RequestHandler):

    def get(self, url):
        sc = SignatureCollection(settings.DB[settings.DEFAULT_COLLECTION])
        d = sc.similarity_search(url)
        self.write({'result': d})

