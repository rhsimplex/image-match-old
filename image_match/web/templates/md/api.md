# {{market}} Similarity Search API

##Introduction

Match images from the web against images in the {{market}} marketplace.  Try it in the browser at [labs.ascribe.io{{BASE_URL}}/{{market}}](http://labs.ascribe.io{{BASE_URL}}/{{market}}/).


## Basics

The base URL is `http://labs.ascribe.io{{BASE_URL}}/{{market}}/search/`

A request using curl looks like:

`$ curl http://labs.ascribe.io{{BASE_URL}}/{{market}}/api/search/?url=http%3A%2F%2Fcdn...com%2Fthumb%2F640%2F480%2....jpg`

<pre>
{
    "url": "http://www.example.org/image.jpg"
    "results": [
        {
            "url": "http://www.example.org/img1.jpg",
            "dist": 0,
            "id": 12
        },
        {
            "url": "http://www.example.org/img2.jpg",
            "dist": 0.43739087511375,
            "id": 528492
        },
        {
            "url": "http://www.example.org/img3.jpg",
            "dist": 0.4438784282276083,
            "id": 27482264
        },
        {
            "url": "http://www.example.org/img4.jpg",
            "dist": 0.4500463733919593,
            "id": 27550153
        },
        {
            "url": "http://www.example.org/img5.jpg",
            "dist": 0.46666245188890754,
            "id": 22654175
        },
        {
            "url": "http://www.example.org/img6.jpg",
            "dist": 0.4687795429136797,
            "id": 30254571
        },
        {
            "url": "http://www.example.org/img7.jpg",
            "dist": 0.4719155672668924,
            "id": 24884288
        },
        {
            "url": "http://www.example.org/img8.jpg",
            "dist": 0.4821427959813869,
            "id": 14152888
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

The body if the reponse is a `JSON` dictionary containing:

- **url**: the URL of the image to match
- **results**: an array of matches containing:
  - **url**: the URL of the matched image
  - **dist**: the distance from the original image
  - **id**: the ID of the image
