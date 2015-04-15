import tornado.web
from image_match.web import settings



class RequestHandler(tornado.web.RequestHandler):

    def get_template_namespace(self):
        ns = super(RequestHandler, self).get_template_namespace()
        print 'hello', settings.BASE_URL
        ns.update({
            'BASE_URL': settings.BASE_URL
        })

        return ns

