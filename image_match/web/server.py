from image_match.web import settings

import tornado.ioloop
import tornado.web
from tornado.options import options
from tornado.log import enable_pretty_logging
from image_match.web.handlers import api, pages


options.define('port', default=8888, help='Port to listen on', type=int)
options.define('debug', default=False, help='Start the server in debug mode',
               type=bool)
options.parse_command_line()
enable_pretty_logging()


application = tornado.web.Application([
    (r'/(favicon/.ico)', tornado.web.StaticFileHandler,
     {'path': settings.FAVICON_ICO}),

    (r'/static/(.*)', tornado.web.StaticFileHandler,
     {'path': settings.STATIC_PATH}),

    (r'/(.+)/docs/(.*)', pages.Documentation),

    (r'/(.+)/api/search/(.*)', api.Search),

    (r'/(.+)', pages.Home),
], debug=options.debug, template_path=settings.TEMPLATE_PATH)


if __name__ == '__main__':
    application.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
