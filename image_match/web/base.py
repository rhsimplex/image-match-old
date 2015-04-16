import tornado.web
import tornado.escape
from image_match.web import settings



class RequestHandler(tornado.web.RequestHandler):

    def get_template_namespace(self):
        ns = super(RequestHandler, self).get_template_namespace()
        ns.update({
            'BASE_URL': settings.BASE_URL,
            'escape': tornado.escape.url_escape
        })

        return ns

