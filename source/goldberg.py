from skimage.color import rgb2gray
from scipy.ndimage import imread
import numpy as np

class ImageSignature(object):
    """Generates an image signature.

    Based on the method of Goldberg, et al. Available at
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf
    """

    def __init__(self):
        pass

    @staticmethod
    def preprocess_image(imagepath):
        """Loads an image and converts to greyscale.

        Corresponds to 'step 1' in Goldberg's paper

        Keyword arguments:
        imagepath -- path to image
        """
        try:
            return rgb2gray(imread(imagepath))
        except IOError:
            print 'File not found: %s' % imagepath

    @staticmethod
    def crop_image(image, lower_percentile=5, upper_percentile=95):
        """Crops an image, removing featureless regions.

        Corresponds to the first part of 'step 2' in Goldberg's paper

        Keyword arguments:
        image -- n x m array
        lower_percentile -- crop image by percentage of difference (default 5)
        upper_percentile -- as lower_percentile (default 95)
        """
        cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=0))      #column-wise differences
        rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))      #row-wise differences

        upper_column_limit = np.searchsorted(cw, np.percentile(cw, upper_percentile))
        lower_column_limit = np.searchsorted(cw, np.percentile(cw, lower_percentile))
        upper_row_limit = np.searchsorted(rw, np.percentile(rw, upper_percentile))
        lower_row_limit = np.searchsorted(rw, np.percentile(rw, lower_percentile))

        return image[lower_row_limit:upper_row_limit, lower_column_limit:upper_column_limit]
        
        
