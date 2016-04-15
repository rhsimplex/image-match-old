import pytest
import urllib
from elasticsearch import Elasticsearch, ConnectionError

from image_match.elasticsearch_driver import SignatureES

test_img_url1 = 'https://camo.githubusercontent.com/810bdde0a88bc3f8ce70c5d85d8537c37f707abe/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f652f65632f4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a70672f36383770782d4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a7067'
test_img_url2 = 'https://camo.githubusercontent.com/826e23bc3eca041110a5af467671b012606aa406/68747470733a2f2f63322e737461746963666c69636b722e636f6d2f382f373135382f363831343434343939315f303864383264653537655f7a2e6a7067'
urllib.urlretrieve(test_img_url1, 'test1.jpg')

es = Elasticsearch()
index_name = 'test_environment'


def test_elasticsearch_running():
    try:
        es.ping()
        assert True
    except ConnectionError:
        pytest.fail('Elasticsearch not running')


def test_add_image_by_url():
    ses = SignatureES(es, index='test_environment')
    ses.add_image(test_img_url1)
    ses.add_image(test_img_url2)
    assert True


def test_add_image_by_path():
    ses = SignatureES(es, index='test_environment')
    ses.add_image('test1.jpg')
    assert True


def test_add_image_as_bytestream():
    ses = SignatureES(es, index='test_environment')
    with open('test1.jpg', 'rb') as f:
        ses.add_image('bytestream_test', img=f.read(), bytestream=True)
    assert True


def test_add_image_with_different_name():
    ses = SignatureES(es, index='test_environment')
    with open('test1.jpg', 'rb') as f:
        ses.add_image('custom_name_test', img='test1.jpg', bytestream=False)
    assert True


def test_lookup_from_url():
    ses = SignatureES(es, index='test_environment')
    r = ses.search_image(test_img_url1)
    assert len(r) == 5


def test_lookup_from_file():
    ses = SignatureES(es, index='test_environment')
    r = ses.search_image('test1.jpg')
    assert len(r) == 5


def test_lookup_from_bytestream():
    ses = SignatureES(es, index='test_environment')
    with open('test1.jpg', 'rb') as f:
        r = ses.search_image(f.read(), bytestream=True)
    assert len(r) == 5


def test_lookup_with_cutoff():
    ses = SignatureES(es, index='test_environment', distance_cutoff=0.30)
    r = ses.search_image('test1.jpg')
    assert len(r) == 4


def check_distance_consistency():
    ses = SignatureES(es, index='test_environment')
    r = ses.search_image('test1.jpg')
    assert r[0]['dist'] == 0.0
    assert r[-1]['dist'] == 0.42672771706789686
