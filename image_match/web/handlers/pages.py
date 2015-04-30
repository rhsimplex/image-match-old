import os
from image_match.web import settings
from image_match.web.base import RequestHandler, SimilaritySearchHandler
import tornado.web
import markdown


class Home(SimilaritySearchHandler):

    def handle_empty_query(self):
        samples = list(self.collection.find().limit(6))
        self.render('home.html',
                    market=self.market,
                    total='{}k'.format(self.collection.count() / 1000),
                    samples=samples)

    def handle_error(self, error):
        self.render('error.html', error=error, market=self.market)

    def handle_response(self, result, image_url, request_time, lookup_time):
        self.render('result.html',
                    result=result,
                    market=self.market,
                    image_url=image_url,
                    request_time=request_time,
                    lookup_time=lookup_time,
                    round=lambda x: round(x, 3))


class Documentation(RequestHandler):

    def get(self, market, name):
        path = os.path.join(settings.TEMPLATE_PATH, 'md', name + '.md')
        try:
            content = open(path).read()
        except IOError:
            raise tornado.web.HTTPError(404)
        template = tornado.web.template.Template(markdown.markdown(content))

        self.render('documentation.html',
                    raw_html=template.generate(market=market),
                    market=market)
