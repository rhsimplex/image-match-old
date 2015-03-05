import tornado.ioloop
import tornado.web
from pymongo import MongoClient
from stilnest_setup import StilnestCollection
import json

client = MongoClient()
db = client['example_db']
c = db['example_collection']


class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json') 

    def get(self, url):
        sc = StilnestCollection(c)
        d = sc.stilnest_lookup(url)
        self.write(json.dumps(d))


application = tornado.web.Application([
    (r'/image_match/(.*)', MainHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
