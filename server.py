import tornado.ioloop
import tornado.web
import tornado.options
from tornado.log import enable_pretty_logging

from pymongo import MongoClient
from bson.json_util import dumps

from stilnest_setup import StilnestCollection


tornado.options.parse_command_line()
enable_pretty_logging()


client = MongoClient()
db = client['stilnest']
c = db['signatures']


class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json') 

    def get(self, url):
        sc = StilnestCollection(c)
        d = sc.stilnest_lookup(url)
        self.write(dumps(d))


application = tornado.web.Application([
    (r'/image_match/(.*)', MainHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
