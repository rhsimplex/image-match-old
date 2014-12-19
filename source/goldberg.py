from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np


class ImageSignature(object):
    """Image signature generator.

    Based on the method of Goldberg, et al. Available at
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf
    """

    def __init__(self, n=9, crop_percentiles=(5, 95), P=None, diagonal_neighbors=True,
                 identical_tolerance=2/255., n_levels=2):
        """Initialize the signature generator.

        The default parameters match those given in Goldberg's paper.

        Keyword arguments:
        n -- size of grid imposed on image. Grid is n x n (default 9)
        
        crop_percentiles -- lower and upper bounds when considering how much
            variance to keep in the image (default (5, 95))
        
        P -- size of sample region, P x P. If none, uses a sample region based
            on the size of the image (default None)

        diagonal_neighbors -- whether to include diagonal neighbors (default True)

        identical_tolerance -- cutoff difference for declaring two adjacent
            grid points identical (default 2/255)

        n_levels -- number of positive and negative groups to stratify neighbor
            differences into. n = 2 -> [-2, -1, 0, 1, 2] (default 2)

        """

        # check inputs
        assert crop_percentiles is None or len(crop_percentiles) == 2,\
            'crop_percentiles should be a two-value tuple, or None'
        if crop_percentiles is not None:
            assert crop_percentiles[0] >= 0,\
                'Lower crop_percentiles limit should be > 0 (%r given)'\
                % crop_percentiles[0]
            assert crop_percentiles[1] <= 100,\
                'Upper crop_percentiles limit should be < 100 (%r given)'\
                % crop_percentiles[1]
            assert crop_percentiles[0] < crop_percentiles[1],\
                'Upper crop_percentile limit should be greater than lower limit.'
            self.lower_percentile = crop_percentiles[0]
            self.upper_percentile = crop_percentiles[1]
            self.crop_percentiles = crop_percentiles
        else:
            self.crop_percentiles = crop_percentiles
            self.lower_percentile = 0
            self.upper_percentile = 100

        assert type(n) is int, 'n should be an integer > 1'
        assert n > 1, 'n should be greater than 1 (%r given)' % n
        self.n = n
        
        assert type(P) is int or P is None, 'P should be an integer >= 1, or None'
        if P is not None:
            assert P >= 1, 'P should be greater than 0 (%r given)' % n
        self.P = P
        
        assert type(diagonal_neighbors) is bool, 'diagonal_neighbors should be boolean'
        self.diagonal_neighbors = diagonal_neighbors
        
        assert type(identical_tolerance) is float or type(identical_tolerance) is int,\
            'identical_tolerance should be a number between 1 and 0'
        assert 0. <= identical_tolerance <= 1.,\
            'identical_tolerance should be greater than zero and less than one (%r given)' % identical_tolerance
        self.identical_tolerance = identical_tolerance

        assert type(n_levels) is int, 'n_levels should be an integer'
        assert n_levels > 0, 'n_levels should be > 0 (%r given)' % n_levels
        self.n_levels = n_levels

    def generate_signature(self, path):
        """Generates an image signature.

        See section 3 of Goldberg, et al.

        Keyword arguments:
        path -- image path

        Returns a signature array
        """

        # Step 1:    Load image as array of grey-levels
        im_array = self.preprocess_image(path)
        
        # Step 2a:   Determine cropping boundaries
        if self.crop_percentiles is not None:
            image_limits = self.crop_image(im_array,
                                           lower_percentile=self.lower_percentile,
                                           upper_percentile=self.upper_percentile)
        else:
            image_limits = None
        
        # Step 2b:   Generate grid centers
        x_coords, y_coords = self.compute_grid_points(im_array,
                                                      n=self.n, window=image_limits)
        
        # Step 3:    Compute grey level mean of each P x P
        #           square centered at each grid point
        avg_grey = self.compute_mean_level(im_array, x_coords, y_coords, P=self.P)

        # Step 4a:   Compute array of differences for each
        #           grid point vis-a-vis each neighbor 
        diff_mat = self.compute_differentials(avg_grey,
                                              diagonal_neighbors=self.diagonal_neighbors)

        # Step 4b: Bin differences to only 2n+1 values
        self.normalize_and_threshold(diff_mat,
                                     identical_tolerance=self.identical_tolerance,
                                     n_levels=self.n_levels)

        # Step 5: Flatten array and return signature
        return np.ravel(diff_mat).astype('int8')

    @staticmethod
    def preprocess_image(imagepath):
        """Loads an image and converts to greyscale.

        Corresponds to 'step 1' in Goldberg's paper

        Keyword arguments:
        imagepath -- path to image
        """
        return rgb2gray(imread(imagepath))

    @staticmethod
    def crop_image(image, lower_percentile=5, upper_percentile=95):
        """Crops an image, removing featureless regions.

        Corresponds to the first part of 'step 2' in Goldberg's paper

        Keyword arguments:
        image -- n x m array
        lower_percentile -- crop image by percentage of difference (default 5)
        upper_percentile -- as lower_percentile (default 95)
        """
        # row-wise differences
        rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=1))
        # column-wise differences
        cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))

        # compute percentiles
        upper_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, upper_percentile),
                                             side='left')
        lower_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, lower_percentile),
                                             side='right')
        upper_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, upper_percentile),
                                          side='left')
        lower_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, lower_percentile),
                                          side='right')
       
        # if image is nearly featureless, use default region
        if lower_row_limit > upper_row_limit:
            lower_row_limit = int(lower_percentile/100.*image.shape[0])
            upper_row_limit = int(upper_percentile/100.*image.shape[0])
        if lower_column_limit > upper_column_limit:
            lower_column_limit = int(lower_percentile/100.*image.shape[1])
            upper_column_limit = int(upper_percentile/100.*image.shape[1])

        return [(lower_row_limit, upper_row_limit),
                (lower_column_limit, upper_column_limit)]
        
    @staticmethod
    def compute_grid_points(image, n=9, window=None):
        """Computes grid points for image analysis.

        Corresponds to the second part of 'step 2' in the paper
        
        Keyword arguments:
        image -- n x m array
        n -- number of gridpoints in each direction (default 9)
        window -- limiting coordinates [(t, b), (l, r)] (default None)
        """

        # if no limits are provided, use the entire image
        if window is None:
            window = [(0, image.shape[0]), (0, image.shape[1])]

        x_coords = np.linspace(window[0][0], window[0][1], n + 2, dtype=int)[1:-1]
        y_coords = np.linspace(window[1][0], window[1][1], n + 2, dtype=int)[1:-1]

        return x_coords, y_coords      # return pairs

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
            P = max([2.0, int(0.5 + min(image.shape)/20.)])     # per the paper

        avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))
        
        for i, x in enumerate(x_coords):        # not the fastest implementation
            lower_x_lim = x - P/2
            upper_x_lim = lower_x_lim + P
            for j, y in enumerate(y_coords):
                lower_y_lim = y - P/2
                upper_y_lim = lower_y_lim + P
                
                avg_grey[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                        lower_y_lim:upper_y_lim])  # no smoothing here as in the paper

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
        right_neighbors = -np.concatenate((np.diff(grey_level_matrix),
                                           np.zeros(grey_level_matrix.shape[0]).
                                           reshape((grey_level_matrix.shape[0], 1))),
                                          axis=1)
        left_neighbors = -np.concatenate((right_neighbors[:, -1:],
                                          right_neighbors[:, :-1]),
                                         axis=1)

        down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0),
                                          np.zeros(grey_level_matrix.shape[1]).
                                          reshape((1, grey_level_matrix.shape[1]))))

        up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

        if diagonal_neighbors:
            # this implementation will only work for a square (m x m) grid
            diagonals = np.arange(-grey_level_matrix.shape[0] + 1,
                                  grey_level_matrix.shape[0])
            
            upper_left_neighbors = sum(
                [np.diagflat(np.insert(np.diff(np.diag(grey_level_matrix, i)), 0, 0), i)
                 for i in diagonals])
            lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:],
                                            (0, 1), mode='constant')
        
            # flip for anti-diagonal differences
            flipped = np.fliplr(grey_level_matrix)
            upper_right_neighbors = sum([np.diagflat(np.insert(
                np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
            lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
                                           (0, 1), mode='constant')

            
            return np.dstack(np.array([
                upper_left_neighbors,
                up_neighbors,
                np.fliplr(upper_right_neighbors),
                left_neighbors,
                right_neighbors,
                np.fliplr(lower_left_neighbors),
                down_neighbors,
                lower_right_neighbors]))
        
        return np.dstack(np.array([
            up_neighbors,
            left_neighbors,
            right_neighbors,
            down_neighbors]))

    @staticmethod
    def normalize_and_threshold(difference_array,
                                identical_tolerance=2/255., n_levels=2):
        """Normalizes difference matrix in place.

        'Step 4' of the paper.

        Keyword arguments:
        difference_array -- n x n x l array, where l are the differences between the grid point and its neighbors
        identical_tolerance -- maximum amount two gray values can differ and still be considered equivalent
        n_levels -- bin differences into 2 n + 1 bins (e.g. n_levels=2 -> [-2, -1, 0, 1, 2])
        """
        
        # set very close values as equivalent
        mask = np.abs(difference_array) < identical_tolerance
        difference_array[mask] = 0.
        
        # if image is essentially featureless, exit here
        if np.all(mask):
            return None

        # bin so that size of bins on each side of zero are equivalent
        positive_cutoffs = np.percentile(difference_array[difference_array > 0.],
                                         np.linspace(0, 100, n_levels+1))
        negative_cutoffs = np.percentile(difference_array[difference_array < 0.],
                                         np.linspace(100, 0, n_levels+1))
            
        for level, interval in enumerate([positive_cutoffs[i:i+2]
                                          for i in range(positive_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array >= interval[0]) &
                             (difference_array <= interval[1])] = level + 1

        for level, interval in enumerate([negative_cutoffs[i:i+2]
                                          for i in range(negative_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array <= interval[0]) &
                             (difference_array >= interval[1])] = -(level + 1)
        
        return None
