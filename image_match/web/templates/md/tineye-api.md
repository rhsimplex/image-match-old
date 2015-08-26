# Similarity Search API

##Introduction

Match images from the web against images in the {{market}} marketplace.  Try it in the browser at [labs.ascribe.io{{BASE_URL}}/{{market}}](http://labs.ascribe.io{{BASE_URL}}/{{market}}/).


## Basics

The base URL is `http://labs.ascribe.io{{BASE_URL}}/{{market}}/search/`

A request using curl looks like:

`$ curl http://labs.ascribe.io{{BASE_URL}}/{{market}}/api/search/?url=http%3A%2F%2Fcdn...com%2Fthumb%2F640%2F480%2....jpg`

<pre>
{
    "url": "http://i.imgur.com/ZFJ228a.jpg",
    "result": [
        {
            "backlink": "http://cera-astronomie.forumactif.com/cinema-et-television-f25/h2d2-le-guide-du-voyageur-galactique-t943.htm",
            "backlinks": [
                {
                    "backlink": "http://cera-astronomie.forumactif.com/cinema-et-television-f25/h2d2-le-guide-du-voyageur-galactique-t943.htm",
                    "crawl_date": "2008-04-24",
                    "url": "http://img62.imageshack.us/img62/6631/hhgttg22ioo1gm.jpg"
                }
            ],
            "height": 318,
            "url": "http://img62.imageshack.us/img62/6631/hhgttg22ioo1gm.jpg",
            "width": 480
        },
        {
            "backlink": "http://artcestralz.canalblog.com/archives/2005/03/28/402584.html",
            "backlinks": [
                {
                    "backlink": "http://artcestralz.canalblog.com/archives/2005/03/28/402584.html",
                    "crawl_date": "2008-04-26",
                    "url": "http://storage.canalblog.com/17/10/20756/10596914.jpg"
                },
                {
                    "backlink": "http://artcestralz.canalblog.com/",
                    "crawl_date": "2008-04-20",
                    "url": "http://storage.canalblog.com/17/10/20756/10596914.jpg"
                }
            ],
            "height": 320,
            "url": "http://storage.canalblog.com/17/10/20756/10596914.jpg",
            "width": 463
        },
        {
            "backlink": "http://www.ohhcrapp.net/2008/03/4-capital-letters-printed-in-gold.html",
            "backlinks": [
                {
                    "backlink": "http://www.ohhcrapp.net/2008/03/4-capital-letters-printed-in-gold.html",
                    "crawl_date": "2009-11-08",
                    "url": "http://bp3.blogger.com/_8Q6k-J9bjqw/R8cMUkIgk6I/AAAAAAAAAAM/6xQP361P-zA/S220/5990698.png"
                }
            ],
            "height": 100,
            "url": "http://bp3.blogger.com/_8Q6k-J9bjqw/R8cMUkIgk6I/AAAAAAAAAAM/6xQP361P-zA/S220/5990698.png",
            "width": 100
        }
    ]
}
</pre>

If an error occurs, the JSON will contain the key ``error``.
<pre>
{
    "error": "Timeout downloading the image"
}
</pre>


# API Documentation

### Version

0.1 Beta

### Roadmap

Features on the roadmap:

 * Rotation
 * Mirroring
 * Inversion is implemented but not yet turned on
 * Cropping resilience will be improved


## GET /search
`http://labs.ascribe.io{{BASE_URL}}/{{market}}/api/search/?url=<url>`

`?url=<url>` should be encoded.


###Description

Retrieves matches to image found at the supplied URL.


### Response

The body of the reponse is a `JSON` dictionary containing:

- **url**: the URL of the image to match
- **results**: an array of matches containing:
  - **url**: the URL of the matched image
  - **height**: the height of the image
  - **width**: the width of the image
  - **backlink**: the first backlink available for the image
  - **backlinks**: an array of backlinks containing:
    - **backlink**: the url of the backlink
    - **crawl_date**: the date the image was crawled
    - **url**: the url of the image
