import tornado.web
from image_match.signature_database import SignatureCollection
from image_match.web import settings


class Search(tornado.web.RequestHandler):

    def get(self, url):
        sc = SignatureCollection(settings.DB[settings.DEFAULT_COLLECTION])
        d = sc.similarity_search(url)
        self.write({'result': d})

