# image-match
image-match is a simple package for finding approximate image matches from a
corpus.  It is similar, for instance, to [pHash](http://www.phash.org/), but
includes a database backend that easily scales to billions of images and
supports sustained high rates of image insertion: up to 10,000 images/s on our
cluster!

Based on the paper [_An image signature for any kind of image_, Goldberg et
al](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps).  There is an existing
[reference implementation](https://www.pureftpd.org/project/libpuzzle) which
may be more suited to your needs.

## Getting started
You'll need a scientific Python distribution and a database backend. Currently
we use Elasticsearch as a backend.


### numpy, PIL, skimage, etc.
Image-match requires several scientific Python packages.  Although they can be
installed and built individually, they are often bundled in a custom Python
distribution, for instance [Anaconda](https://www.continuum.io/why-anaconda).
Installation instructions can be found
[here](https://www.continuum.io/downloads#_unix).


### Elasticsearch
If you just want to generate and compare image signatures, you can skip this
step. If you want to search over a corpus of millions or billions of image
signatures, you will need a database backend. We built image-match around
[Elasticsearch](https://www.elastic.co/).  See download and installation
instructions [here](https://www.elastic.co/downloads/elasticsearch).


### Install image-match
1. Clone this repository:

  ```
  $ git clone https://github.com/ascribe/image-match.git
  ```

2. Install image-match

  ```
  $ pip install numpy
  $ pip install .
  ```

3. Make sure elasticsearch is running (optional):

  For example, on Ubuntu you can check with:

  ```
  $ sudo service elasticsearch status
  ```

  If it's not running, simply run:

  ```
  $ sudo service elasticsearch start
  ```

## Image signatures and distances
Consider these two photographs of the [Mona
Lisa](https://en.wikipedia.org/wiki/Mona_Lisa):

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg)

(credit:
[Wikipedia](https://en.wikipedia.org/wiki/Mona_Lisa#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg)
Public domain)

![](https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg)

(credit:
[WikiImages](https://pixabay.com/en/mona-lisa-painting-art-oil-painting-67506/)
Public domain)

Though it's obvious to any human observer that this is the same image, we can
find a number of subtle differences: the dimensions, palette, lighting and so
on are different in each image. image-match will give us numerical comparison:

```python
from image_match.goldberg import ImageSignature
gis = ImageSignature()
a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
gis.normalized_distance(a, b)
```

Returns `0.22095170140933634`. Normalized distances of less than `0.40` are
very likely matches. If we try this again against a dissimilar image, say,
Caravaggio's [Supper at
Emmaus](https://en.wikipedia.org/wiki/Supper_at_Emmaus_(Caravaggio),_London):
![](https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg)

(credit: [Wikipedia](https://en.wikipedia.org/wiki/Caravaggio#/media/File:Caravaggio_-_Cena_in_Emmaus.jpg) Public domain)

against one of the Mona Lisa photographs:
```python
c = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
gis.normalized_distance(a, c)
```

Returns `0.68446275381507249`, almost certainly not a match.  image-match
doesn't have to generate a signature from a URL; a file-path or even an
in-memory bytestream will do (be sure to specify `bytestream=True` in the
latter case).

Now consider this subtly-modified version of the Mona Lisa:

![https://www.flickr.com/photos/planetrussell/6814444991](https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg)

(credit: [Michael Russell](https://www.flickr.com/photos/planetrussell/6814444991) [Attribution-ShareAlike 2.0 Generic](https://creativecommons.org/licenses/by-sa/2.0/))

How similar is it to our original Mona Lisa?
```python
d = gis.generate_signature('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')
gis.normalized_distance(a, d)
```

This gives us `0.42557196987336648`. So markedly different than the two
original Mona Lisas, but considerably closer than the Caravaggio.


## Storing and searching the Signatures
In addition to generating image signatures, image-match also facilitates
storing and efficient lookup of images—even for up to (at least) a billion
images.  Instagram account only has a few million images? Don't worry, you can
get 80M images [here](http://horatio.cs.nyu.edu/mit/tiny/data/index.html]) to
play with.

A signature database wraps an Elasticsearch index, so you'll need Elasticsearch
up and running. Once that's done, you can set it up like so:

```python
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES

es = Elasticsearch()
ses = SignatureES(es)
```

By default, the Elasticsearch index name is "images" and the document type
"image," but you can change these via the `index` and `doc_type` parameters.

Now, let's store those pictures from before in the database:

```python
ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
ses.add_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
ses.add_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')
```

Now let's search for one of those Mona Lisas:

```python
ses.search_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
```

The result is a list of hits:

```python
[
 {'dist': 0.0,
  'id': u'AVM37oZq0osmmAxpPvx7',
  'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
  'score': 7.937254},
 {'dist': 0.22095170140933634,
  'id': u'AVM37nMg0osmmAxpPvx6',
  'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
  'score': 0.28797293},
 {'dist': 0.42557196987336648,
  'id': u'AVM37p530osmmAxpPvx9',
  'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
  'score': 0.0499953}
]
```

`dist` is the normalized distance, like we computed above. Hence, lower numbers
are better with `0.0` being a perfect match. `id` is an identifier assigned by
the database. `score` is computed by Elasticsearch, and higher numbers are
better here. `path` is the original path (url or file path).

Notice all three Mona Lisa images appear in the results, with the identical
image being a perfect (`'dist': 0.0`) match. If we search instead for the
Caravaggio,

```python
ses.search_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
```

You get:

```python
[
 {'dist': 0.0,
  'id': u'AVMyXQFw0osmmAxpPvxz',
  'path': u'https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg',
  'score': 7.937254}
]
```

It only finds the Caravaggio, which makes sense! But what if we wanted an even
more restrictive search? For instance, maybe we only want unmodified Mona Lisas
-- just photographs of the original. We can restrict our search with a hard
cutoff using the `distance_cutoff` keyword argument:

```python
ses = SignatureES(es, distance_cutoff=0.3)
ses.search_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
```

Which now returns only the unmodified, catless Mona Lisas:

```python
[
 {'dist': 0.0,
  'id': u'AVMyXOz30osmmAxpPvxy',
  'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
  'score': 7.937254},
 {'dist': 0.23889600350807427,
  'id': u'AVMyXMpV0osmmAxpPvxx',
  'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
  'score': 0.28797293}
]
```

### Distorted and transformed images

image-match is also robust against basic image transforms. Take this squashed
Mona Lisa:

![](http://i.imgur.com/CVYBCCy.jpg)

No problem, just search as usual:

```python
ses.search_image('http://i.imgur.com/CVYBCCy.jpg')
```

returns

```
[
 {'dist': 0.15454905655638429,
  'id': u'AVM37oZq0osmmAxpPvx7',
  'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
  'score': 1.6818419},
 {'dist': 0.24980626832071956,
  'id': u'AVM37nMg0osmmAxpPvx6',
  'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
  'score': 0.16198477},
 {'dist': 0.43387141782958921,
  'id': u'AVM37p530osmmAxpPvx9',
  'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
  'score': 0.031996995}
]
```

as expected.  Now, consider this rotated version:

![](http://i.imgur.com/T5AusYd.jpg)

image-match doesn't search for rotations and mirror images by default.
Searching for this image will return no results, unless you search with
`all_orientations=True`:

```python
ses.search_image('http://i.imgur.com/T5AusYd.jpg', all_orientations=True)
```

Then you get the expected matches.


## Other database backends
Though we designed image-match with Elasticsearch in mind, other database
backends are possible. For demonstration purposes we include also a
[MongoDB](https://www.mongodb.org/) driver:

```python
from image_match.mongodb_driver import SignatureMongo
from pymongo import MongoClient

client = MongoClient(connect=False)
c = client.images.images

ses = SignatureMongo(c)
```

now you can use the same functionality as above like `ses.add_image(...)`.

We tried to separate signature logic from the database insertion/search as much
as possible.  To write your own database backend, you can inherit from the
`SignatureDatabaseBase` class and override the appropriate methods:

```python
from signature_database_base import SignatureDatabaseBase
# other relevant imports


class MySignatureBackend(SignatureDatabaseBase):

    # if you need to do some setup, override __init__
    def __init__(self, myarg1, myarg2, *args, **kwargs):
        # do some initializing stuff here if necessary
        # ...
        super(MySignatureBakend, self).__init__(*args, **kwargs)

    # you MUST implement these two functions
    def search_single_record(self, rec):
        # should query your database given a record generated from signature_database_base.make_record
        # ...
        # should return a list of dicts like [{'id': 'some_unique_id_from_db', 'dist': 0.109234, 'path': 'url/or/filepath'}, {...} ...]
        # you can have other keys, but you need at least id and dist
        return formatted_results

    def insert_single_record(self, rec):
        # if your database driver or instance can accept a dict as input, this should be very simple

    # ...
```

Unfortunately, implementing a good `search_single_record` function does require
some knowledge of [the search
algorithm](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps). You can also look at
the two included database drivers for guidelines.

