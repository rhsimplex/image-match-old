import requests
import time

"""
Similarity search implemented via the tineye API.

Uses the requests library: http://docs.python-requests.org/en/latest/

"""

URL = 'http://api.tineye.com/rest/search/'
public_key = 'dT-Q98^ICh-H8AboWSMQ'
private_key_path = 'private_key.txt'


def similarity_search(path, limit=10):
    with open(private_key_path) as f:
        args = {
            'public key': public_key,
            'private key': f.read(),
            'limit': str(limit),
            'image data filename': path,
            'date': str(int(time.time()))
        }

        files = {'file': open(path, 'rb')}

        r = requests.post(URL, data=args, files=files)