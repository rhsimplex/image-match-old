# EyeEm Similarity Search API

##Introduction
Match images from the web against images in the EyeEm marketplace.  Try it in the browser at xxxxxxxx

## API Documentation
### Endpoints
  * ** [Search](endpoints/similarity_search#files)**

## Basics
***
The base URL is `http://CHANGETHIS/search/`

A request using curl looks like:

`$ curl http://CHANGETHIS/search/http://cdn.eyeem.com/thumb/640/480/12683532893904596226_26bbd8936e_o.jpg`

The ```dist``` field indicates how closely matched the images are. A perfect match has a ```dist``` value of ```0.0```.

<pre>
{
    "result": [
        {
            "path": "http://www.eyeem.com/thumb/640/480/12683532893904596226_26bbd8936e_o.jpg",
            "dist": 0,
            "id": 12
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/de05a37ae69d9a2a8280ae2c243c2645e2046d2c-1337985253",
            "dist": 0.43739087511375,
            "id": 528492
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/491e4385482726023f0d60e3e2ba8bfc93c6bf9d-1389497502",
            "dist": 0.4438784282276083,
            "id": 27482264
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/07a9d6a0c422363f2570c3b973c23a7c6fb131cf-1389573626",
            "dist": 0.4500463733919593,
            "id": 27550153
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/7ce63ce7a4e355bf98d431d23e9d515b9c0d3f3d-1382519368",
            "dist": 0.46666245188890754,
            "id": 22654175
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/3ba34f3934d84f8fdcaa306044b33a5c4583c1d2-1392830857",
            "dist": 0.4687795429136797,
            "id": 30254571
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/bcf73e104a60e046341a45a75d25bc0ae5b2c338-1385909727",
            "dist": 0.4719155672668924,
            "id": 24884288
        },
        {
            "path": "http://www.eyeem.com/thumb/640/480/26151dd2148bebc36c9d64113fb4cc1bb10e7431-1370539530",
            "dist": 0.4821427959813869,
            "id": 14152888
        }
    ]
}
</pre>

### Version
***0.1 Beta***

Not yet implemented:
* Rotation, mirroring, inversion is implemented but not yet turned on
* Cropping resilience will be improved

# GET /search
`/search`

###Description
***
Retrieves matches to image found at the supplied URL

### Parameters
***
|parameter| description| type| required? |default|
|:---------|:--------------|:----------:|:------------:|:------------:|
||URL to image | string | yes | none

### Response
***

200 and an array of ```{dist, path, id}``` image descriptions


