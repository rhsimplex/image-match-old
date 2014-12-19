"""
Hi Dimi,

You need to have a MongoDB server running and the python API (pymongo) installed.

You can run this script to see if everything is working. I've annotated it line by line
to show you how it works.
"""

"""
If the import statement works you should be golden:
"""
from pymongo import MongoClient

"""
In general, you will need to access or create a MongoDB database. The syntax is the same:
"""

# connect to the server. If your server isn't running, this will fail.
# You can start the server at the command line with $mongod
client = MongoClient()

# access or create a database -- the syntax is the same, and you can call it whatever you like
db = client['example_db']

# access or create a collection. Why MongoDB has two levels of abstraction, I couldn't tell you
c = db['example_collection']

"""
Now you can pass the collection to my library, and start inserting and searching images
"""

from image_match.signature_database import SignatureCollection

# instantiate a SignatureCollection object from the MongoDB collection
sc = SignatureCollection(c)

# let's insert an image
sc.add_image('http://www.bilderbuch-berlin.net/bilder/berlin_mitte_berliner_fernsehturm_architektur_fassade_d277316530_978x1304xin.jpeg')

"""
Let's match it against the same, similar, and different images
"""

# matching against itself. Should print a dict with format:
# {'verdict': 'fail', 'reason': [{similarity_measure, url, unique_MongoDB_id}]}
#
print sc.similarity_search('http://www.bilderbuch-berlin.net/bilder/berlin_mitte_berliner_fernsehturm_architektur_fassade_d277316530_978x1304xin.jpeg')

# matching against itself a very similar image. Should print a dict with format:
# {'verdict': 'pending', 'reason': [{similarity_measure, url, unique_MongoDB_id}]}
#
# If there were many close images, it would return all of them
#
print sc.similarity_search('http://upload.wikimedia.org/wikipedia/commons/b/bf/Berlin_Fernsehturm_2005.jpg')

# matching against an obviously different image. Should print dict with format
# {'verdict': 'pass', 'reason': []}
#
print sc.similarity_search('http://edge.pokerzeit.com/assets/de_DE/photos/berlinfernsehturm-37990.jpg')

"""
Drop the database, if you want.
"""
client.drop_database(db)
