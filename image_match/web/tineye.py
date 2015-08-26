from pytineye import TinEyeAPIRequest


class TinEyeJSONRequest(TinEyeAPIRequest):
    def search_url(
            self, url, offset=0, limit=100, sort='score',
            order='desc', **kwargs):
        """
        Perform searches on the TinEye index using an image URL.

        - `url`, the URL of the image that will be searched for, must be urlencoded.
        - `offset`, offset of results from the start, defaults to 0.
        - `limit`, number of results to return, defaults to 100.
        - `sort`, sort results by score, file size (size), or crawl date (crawl_date),
          defaults to descending (desc).
        - `order`, sort results by ascending (asc) or descending criteria.
        - `kwargs`, to pass extra arguments intended for debugging.

        Returns: a python dictionary.
        """

        params = {
            'image_url': url,
            'offset': offset,
            'limit': limit,
            'sort': sort,
            'order': order}

        obj = self._request('search', params, **kwargs)
        matches = obj['results']['matches']
        rs = []
        for m in matches:
            backlink = m['backlinks'][0]
            parsed = {'url': backlink['url'],
                      'backlink': backlink['backlink'],
                      'backlinks': m['backlinks'],
                      'height': m['height'],
                      'width': m['width']}
            rs.append(parsed)

        return {'url': url, 'results': rs}
