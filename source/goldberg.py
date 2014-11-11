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
        rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=1))      #row-wise differences
        cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))      #column-wise differences

        upper_column_limit = np.searchsorted(cw, np.percentile(cw, upper_percentile), side='left')      #compute percentiles
        lower_column_limit = np.searchsorted(cw, np.percentile(cw, lower_percentile), side='right')
        upper_row_limit = np.searchsorted(rw, np.percentile(rw, upper_percentile), side='left')
        lower_row_limit = np.searchsorted(rw, np.percentile(rw, lower_percentile), side='right')

        return [(lower_row_limit,upper_row_limit), (lower_column_limit,upper_column_limit)]
        
    @staticmethod
    def compute_grid_points(image, n=9, window=None):
        """Computes grid points for image analysis.

        Corresponds to the second part of 'step 2' in the paper
        
        Keyword arguments:
        image -- n x m array
        n -- number of gridpoints in each direction (default 9)
        window -- limiting coordinates [(t, b), (l, r)] (default None)
        """
        if window is None:
            window = [(0, image.shape[0]), (0, image.shape[1])]     #if no limits are provided, use the entire image

        x_coords = np.linspace(window[0][0], window[0][1], n + 2, dtype=int)[1:-1]
        y_coords = np.linspace(window[1][0], window[1][1], n + 2, dtype=int)[1:-1]

        return x_coords, y_coords      #return pairs

    @staticmethod
    def compute_mean_level(image, x_coords, y_coords, P=None):
        """Computes array of greyness means.

        Corresponds to 'step 3'

        Keyword arguments:
        image -- n x m array
        x_coords -- 1d array of row numbers
        y_coords -- 1d array of column numbers
        P -- size of boxes in pixels (default None)
        """

        if P is None:
            P = max([2.0, int(0.5 + min(image.shape)/20.)])     #per the paper

        avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))
        
        for i, x in enumerate(x_coords):        #not the fastest implementation
            lower_x_lim = x - P/2
            upper_x_lim = lower_x_lim + P
            for j, y in enumerate(y_coords):
                lower_y_lim = y - P/2
                upper_y_lim = lower_y_lim + P
                
                avg_grey[i,j] = np.mean(image[lower_x_lim:upper_x_lim, lower_y_lim:upper_y_lim])  #no smoothing here as in the paper

        return avg_grey

                
                

        
