# README #

Image match application based on Goldberg, et al: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.2585

For now, how to get a test database running:

### Dependencies ###
* Python 2.7x
* [MongoDB](http://www.mongodb.org/) and the Python API, [Pymongo](http://api.mongodb.org/python/current/)
* Numpy, skimage

### How do I get set up? ###
Fast demo:
1. Clone repository
2. Run example.py. The comments explain basic usage.

Longer demo:
1. Clone repository
2. Run setup script: `python demo_setup.py`. This may take awhile, especially downloading the images.
3. Run the ipython notebook: `ipython notebook image_match_demo.ipynb`

Let me know of any issues.

-Ryan