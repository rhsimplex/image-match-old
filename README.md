# image-match

image-match is a simple package for finding approximate matches of images  based on Goldberg, et al: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.2585

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

## Examples