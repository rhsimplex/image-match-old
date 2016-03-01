from signature_database_base import SignatureDatabaseBase
from signature_database_base import normalized_distance
from operator import itemgetter
from datetime import datetime
import numpy as np


class SignatureES(SignatureDatabaseBase):

    def __init__(self, es, index='images', doc_type='image', timeout=10, size=100,
                 *args, **kwargs):
        self.es = es
        self.index = index
        self.doc_type = doc_type
        self.timeout = timeout
        self.size = size

        super(SignatureES, self).__init__(*args, **kwargs)

    def search_single_record(self, rec):
        path = rec.pop('path')
        signature = rec.pop('signature')

        fields = ['path', 'signature']

        # build the 'should' list
        should = [{'term': {word: rec[word]}} for word in rec]
        res = self.es.search(index=self.index,
                              doc_type=self.doc_type,
                              body={'query':
                                      {
                                          'filtered': {
                                              'query': {
                                                    'bool': {'should': should}
                                              }
                                          }
                                      }},
                              fields=fields,
                              size=self.size,
                              timeout=self.timeout)['hits']['hits']

        sigs = np.array([x['fields']['signature'] for x in res], dtype='uint8')

        if sigs.size == 0:
            return []

        dists = normalized_distance(sigs, np.array(signature, dtype='uint8'))

        formatted_res = [{'id': x['_id'],
                          'score': x['_score'],
                          'path': x['fields'].get('url', x['fields'].get('path'))[0]}
                         for x in res]

        for i, row in enumerate(formatted_res):
            row['dist'] = dists[i]
        formatted_res = filter(lambda y: y['dist'] < self.distance_cutoff, formatted_res)

        return formatted_res

    def insert_single_record(self, rec):
        rec['timestamp'] = datetime.now()
        self.es.index(index=self.index, doc_type=self.doc_type, body=rec)
