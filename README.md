# image-match

image-match is a simple package for finding approximate image matches from a corpus.  It is similar, for instance, to [pHash](http://www.phash.org/), but includes a database backend that easily scales to billions of images and supports sustained high rates of image insertion: up to 10,000 images/s on our cluster!

Based on the paper [_An image signature for any kind of image_, Goldberg et al](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps).  There is an existing [reference implementation](https://www.pureftpd.org/project/libpuzzle) which may be more suited to your needs.

## Getting started

You'll need a scientific Python distribution and a database backend. Currently we use Elasticsearch as a backend.

### numpy, PIL, skimage, etc.

Image-match requires several scientific Python packages.  Although they can be installed and built individually, they are often bundled in a custom Python distribution, for instance [Anaconda](https://www.continuum.io/why-anaconda). Installation instructions can be found [here](https://www.continuum.io/downloads#_unix).

### Elasticsearch

If you just want to generate and compare image signatures, you can skip this step. If you want to search over a corpus of millions or billions of image signatures, you will need a database backend. We built image-match around [Elasticsearch](https://www.elastic.co/).  See download and installation instructions [here](https://www.elastic.co/downloads/elasticsearch).

### Install image-match
1. Clone this repository:

  ```text
  $ git clone https://github.com/ascribe/image-match.git
  ```

2. Install image-match

  ```text
  $ pip install -r requirements.txt
  $ pip install .
  ```

3. Make sure elasticsearch is running (optional):

  For example, on Ubuntu you can check with:

  ```text
  $ sudo service elasticsearch status
  ```

  If it's not running, simply run:

  ```text
  $ sudo service elasticsearch start
  ```

## Image signatures and distances
Consider these two photographs of the [Mona Lisa](https://en.wikipedia.org/wiki/Mona_Lisa):

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg)

(credit: [Wikipedia](https://en.wikipedia.org/wiki/Mona_Lisa#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg) Public domain)

![](https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg)

(credit: [WikiImages](https://pixabay.com/en/mona-lisa-painting-art-oil-painting-67506/) Public domain)

Though it's obvious to any human observer that this is the same image, we can find a number of subtle differences: the dimensions, palette, lighting and so on are different in each image. image-match will give us numerical comparison:
```python
from image_match.goldberg import ImageSignature
gis = ImageSignature()
a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
gis.normalized_distance(a, b)
```

Returns `0.22095170140933634`. Normalized distances of less than `0.40` are very likely matches. If we try this again against a dissimilar image, say, Caravaggio's [Supper at Emmaus](https://en.wikipedia.org/wiki/Supper_at_Emmaus_(Caravaggio),_London):
![](https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg)

(credit: [Wikipedia](https://en.wikipedia.org/wiki/Caravaggio#/media/File:Caravaggio_-_Cena_in_Emmaus.jpg) Public domain)

against one of the Mona Lisa photographs:
```python
c = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
gis.normalized_distance(a, c)
```

Returns `0.68446275381507249`, almost certainly not a match.  image-match doesn't have to generate a signature from a URL; a file-path or even an in-memory bytestream will do (be sure to specify `bytestream=True` in the latter case).

Now consider this subtly-modified version of the Mona Lisa:

![https://www.flickr.com/photos/planetrussell/6814444991](https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg)

(credit: [Michael Russell](https://www.flickr.com/photos/planetrussell/6814444991) [Attribution-ShareAlike 2.0 Generic](https://creativecommons.org/licenses/by-sa/2.0/))

How similar is it to our original Mona Lisa?
```python
d = gis.generate_signature('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')
gis.normalized_distance(a, d)
```

This gives us `0.42557196987336648`. So markedly different than the two original Mona Lisas, but considerably closer than the Caravaggio.

## Storing the signatures

In addition to generating image signatures, image-match also facilitates storing and efficient lookup of images—even for up to (at least) a billion images.  Instagram account only has a few million images? Don't worry, you can get 80M images [here](http://horatio.cs.nyu.edu/mit/tiny/data/index.html]) to play with.

A signature database wraps an Elasticsearch index, so you'll need Elasticsearch up and running. Once that's done, you can set it up like so:

```python
from elasticsearch import Elasticsearch
from image_match.signature_database import SignatureES

es = Elasticsearch()
ses = SignatureES(es)
```

By default, the Elasticsearch index name is "images" and the document type "image," but you can change these via the `index` and `doc_type` parameters.

Now, let's store those pictures from before in the database:

```python
ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
ses.add_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
ses.add_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')
```

Now let's search for one of those Mona Lisas:

```python
ses.bool_query('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
```

`bool_query` is so-named because it uses [Bool Query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) functionality of Elasticsearch.  We can actually search here without computing signatures, which makes it very fast on large image corpora.

The result is a list of hits:

```python
[
 {'dist': 0.0,
  'id': u'AVMdCX8WrUjAIrQRJPr6',
  'score': 7.937254,
  'url': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
 {'dist': 0.23889600350807427,
  'id': u'AVMdCTpmrUjAIrQRJPr5',
  'score': 0.28797293,
  'url': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
 {'dist': 0.36426714382512343,
  'id': u'AVMdCc9mrUjAIrQRJPr8',
  'score': 0.0499953,
  'url': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
]
```

`dist` is the normalized distance, like we computed above. Hence, lower numbers are better with `0.0` being a perfect match. `id` is an identifier assigned by the database. `score` is computed by Elasticsearch, and higher numbers are better here. `url` is the original path.

Notice all three Mona Lisa images appear in the results, with the identical image being a perfect (`'dist': 0.0`) match. If we search instead for the Caravaggio,

```python
ses.bool_query('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
```

You get:

```python
[
 {'dist': 0.0,
  'id': u'AVMdCaXerUjAIrQRJPr7',
  'score': 7.937254,
  'url': u'https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg'},
 {'dist': 0.48742886583201611,
  'id': u'AVMdCc9mrUjAIrQRJPr8',
  'score': 0.0019998122,
  'url': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
]
```

It catches one of the the Mona Lisas, but at a distance of `0.48742886583201611` it's a very marginal match. We can restrict our search with the `distance_cutoff` parameter:

```python
ses = SignatureES(es, distance_cutoff=0.4)
ses.bool_query('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
```

Which now returns only the identical match.