import tornado.escape
from image_match.signature_database import SignatureCollection
from image_match.web import settings
from image_match.web.base import RequestHandler



class Home(RequestHandler):

    def get(self):
        image_url = self.get_argument('image_url', None)
        collection = settings.DB[settings.DEFAULT_COLLECTION]

        if image_url:
            sc = SignatureCollection(collection)
            d = sc.similarity_search(image_url)
            self.render('result.html', result=d, image_url=image_url, escape=tornado.escape.url_escape)
        else:
            self.render('home.html', total=collection.count())

