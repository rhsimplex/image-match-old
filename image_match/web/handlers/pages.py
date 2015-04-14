import tornado.web


class Home(tornado.web.RequestHandler):

    def get(self, url):
        #sc = SignatureCollection(settings.DB[settings.DEFAULT_COLLECTION])
        #d = sc.similarity_search(url)
        self.render('home.html')

