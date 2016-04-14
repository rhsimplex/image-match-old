import pytest

def test_import():
    from image_match.goldberg import ImageSignature
    gis = ImageSignature()
    assert not gis.P
