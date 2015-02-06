from pytineye import TinEyeAPIRequest
"""
Similarity search implemented via the tineye API.

Uses the pytineye library: https://github.com/TinEye/pytineye

Note: local system time must be more or less accurate for this to work, which
can be an issue on the VMs. If all else fails, use 'sudo date -s JAN 2015 18:00:00'
with the current date and time to fix it.  You don't have to be accurate to the second.

"""

URL = 'http://api.tineye.com/rest/'
public_key = 'dT-Q98^ICh-H8AboWSMQ'
private_key_path = 'private_key.txt'


def similarity_search(path, limit=10):
    with open(private_key_path) as f:
        with open(path, 'rb') as img:
            private_key = f.read()

            api = TinEyeAPIRequest(URL, public_key, private_key)

            response = api.search_data(img.read(), limit=limit)

            if response.matches:
                return {'verdict': 'fail',
                        'reason': [match.backlinks[0].url for match in response.matches]}
            else:
                return {'verdict': 'pass',
                        'reason': []}
