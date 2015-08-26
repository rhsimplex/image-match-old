from image_match.web.base import SimilaritySearchHandler, TineyeSearchHandler


class Search(SimilaritySearchHandler):

    def handle_error(self, error):
        self.write({
            'error': 'Timeout downloading the image',
        })
        self.finish()

    def handle_response(self, result, request_time, lookup_time):
        self.write(result)
        self.finish()


class TineyeSearch(TineyeSearchHandler):

    def handle_error(self, error):
        self.write({
            'error': 'Timeout downloading the image',
        })
        self.finish()

    def handle_response(self, result, request_time, lookup_time):
        self.write(result)
        self.finish()
