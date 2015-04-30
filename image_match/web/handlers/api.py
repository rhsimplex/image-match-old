from image_match.web.base import SimilaritySearchHandler


class Search(SimilaritySearchHandler):

    def get(self, market, url):
        self.image_url = url
        super(Search, self).get(market)

    def handle_error(self, error):
        self.write({'error': 'Timeout downloading the image'})
        self.finish()

    def handle_response(self, result, image_url, request_time, lookup_time):
        self.write({'result': result})
        self.finish()
