import numpy as np


class Params:
    """Holding parameters of a 1-dimensional boxplot:
        - median
        - 25/75 percentiles
        - IQR i.e. inter-quantile range
        - whiskers
        - outliers
    """

    def _init__(self, whisker_type='IQR'):
        """
        :param whisker_type : either IQR or MinMax
        """
        if whisker_type not in ['IQR', 'MinMax']:
            raise Exception("Only whisker types 'IQR' and 'MinMax' allowed")
        self.whisker_type = whisker_type
        self.median = None
        self.percentile_low = None
        self.percentile_high = None
        self.iqr = None
        self.min = None
        self.max = None
        self.outliers = None
        self.whiskers = None
        self.upper_limit = None
        self.lower_limit = None
        self.data = None

    def calculate_params(self, data):
        """Calculates all boxplot parameters.
        :param data: one-dimensional data in either list or np.array format
        """
        self.data = data
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        self.calculate_median()
        self.calculate_percentiles()
        self.calculate_iqr()
        self.calculate_min_max()
        self.calculate_whiskers_iqr()

    def calculate_median(self):
        """Calculates median.
        """
        self.median = np.median(self.data)

    def calculate_percentiles(self):
        """Calculates 25 and 75 percentiles.
        """
        self.percentile_low = np.percentile(self.data, 25)
        self.percentile_high = np.percentile(self.data, 75)

    def calculate_iqr(self):
        """Calculates inter-quantile range.
        """
        self.iqr = self.percentile_high - self.percentile_low

    def calculate_min_max(self):
        """Calculates min/max.
        """
        self.min = np.min(self.data)
        self.max = np.max(self.data)

    def calculate_whiskers_iqr(self):
        """Calculates whiskers based on IQR. The upper extreme whisker is
        data-point with largest value within 75th percentile + 1.5*IQR.
        The lower extreme whisker data-point with lowest value within
        25th percentile - 1.5*IQR.
        """
        self.lower_limit = self.percentile_low - 1.5*self.iqr
        self.upper_limit = self.percentile_high + 1.5*self.iqr
        min_idx = self.data >= self.lower_limit
        max_idx = self.data <= self.upper_limit
        out_idx = np.logical_not(np.logical_or(min_idx, max_idx))
        lower_whisker = np.min(self.data[min_idx])
        upper_whisker = np.max(self.data[max_idx])
        self.whiskers = np.array([lower_whisker, upper_whisker])
        self.outliers = self.data[out_idx]

    def _repr__(self):
        output_str = ''
        output_str += f'Median          : {self.median} \n'
        output_str += f'Low percentile  : {self.percentile_low} \n'
        output_str += f'High percentile : {self.percentile_high} \n'
        output_str += f'IQR             : {self.iqr} \n'
        output_str += f'Min             : {self.min} \n'
        output_str += f'Max             : {self.max} \n'
        output_str += f'Whiskers        : {self.whiskers} \n'
        output_str += f'Upper limit     : {self.upper_limit} \n'
        output_str += f'Lower limit     : {self.lower_limit} \n'
        return output_str
