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

    @staticmethod
    def compute_differentials(grey_level_matrix,  diagonal_neighbors=True):
        """Computes differences in greylevels for neighboring grid points.

        First part of 'step 4' in the paper.

        Returns n x n x 8 array for an n x n grid (if diagonal_neighbors == True)

        The n x nth coordinate corresponds to a grid point.  The eight values are
        the differences between neighboring grid points, in this order:

        right
        left
        up
        down
        upper left diagonal
        lower right diagonal
        upper right diagonal
        lower left diagonal

        Keyword arguments:
        grey_level_matrix -- grid of values sampled from image
        diagonal_neighbors -- whether or not to use diagonal neighbors (default True)
        """
        right_neighbors = -np.concatenate((np.diff(grey_level_matrix), np.zeros(grey_level_matrix.shape[0]).reshape((grey_level_matrix.shape[0],1))), axis=1)
        left_neighbors = -np.concatenate((right_neighbors[:, -1:], right_neighbors[:, :-1]), axis=1)

        down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0), np.zeros(grey_level_matrix.shape[1]).reshape((1, grey_level_matrix.shape[1]))))
        up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

        if diagonal_neighbors:
            diagonals = np.arange(-grey_level_matrix.shape[0] + 1, grey_level_matrix.shape[0])      #this implementation will only work for a square (m x m) grid
            
            upper_left_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(grey_level_matrix, i)), 0, 0), i) for i in diagonals])
            lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:], (0, 1), mode='constant')
        
            flipped = np.fliplr(grey_level_matrix)      #flip for anti-diagonal differences
            upper_right_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
            lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:], (0, 1), mode='constant')

            
        return np.dstack(np.array([right_neighbors, left_neighbors, up_neighbors, down_neighbors, upper_left_neighbors, lower_right_neighbors, np.fliplr(upper_right_neighbors), np.fliplr(lower_left_neighbors)]))

    @staticmethod
    def normalize_and_threshold(difference_array, identical_tolerance=2/255., n_levels=2):
        """Normalizes difference matrix in place.

        'Step 4' of the paper.

        Keyword arguments:
        difference_array -- n x n x l array, where l are the differences between the grid point and its neighbors
        identical_tolerance -- maximum amount two gray values can differ and still be considered equivalent
        n_levels -- bin differences into 2 n + 1 bins (e.g. n_levels=2 -> [-2, -1, 0, 1, 2])
        """
        difference_array[np.abs(difference_array) < identical_tolerance] = 0.       #set very close values as equivalent

        positive_cutoffs = np.percentile(difference_array[difference_array > 0.], np.linspace(0, 100, n_levels+1))      #bin so that size of bins on each side of zero are equivalent 
        negative_cutoffs = np.percentile(difference_array[difference_array < 0.], np.linspace(100, 0, n_levels+1))

        for level, interval in enumerate([positive_cutoffs[i:i+2] for i in range(positive_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array >= interval[0]) & (difference_array <= interval[1])] = level + 1

        for level, interval in enumerate([negative_cutoffs[i:i+2] for i in range(negative_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array <= interval[0]) & (difference_array >= interval[1])] = -(level + 1)
        
        return None
