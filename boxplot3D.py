import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle


class BoxplotParams:
    def __init__(self, x, whisker_type='IQR'):
        if not isinstance(x, np.array):
            self._x = np.array(x)
        else:
            self._x = x
        if whisker_type not in ['IQR', 'MinMax']:
            raise Exception("Only whisker types 'IQR' and 'MinMax' allowed")
        self._whisker_type = whisker_type
        self.median = None
        self.perc_25 = None
        self.perc_75 = None
        self._iqr = None
        self._min = None
        self._max = None
        self.outliers = None
        self.whiskers = None

    def _calc_median(self):
        self.median = np.median(self._x)

    def _calc_percentiles(self):
        self.perc_25 = np.percentile(self._x, 25)
        self.perc_75 = np.percentile(self._x, 75)

    def _calc_iqr(self):
        self._iqr = self.perc_75 - self.perc_25

    def _calc_min_max(self):
        self._min = np.min(self._x)
        self._max = np.max(self._x)

    def _calc_whiskers(self):
        lower_limit = self.perc_25 - 1.5*self._iqr
        upper_limit = self.perc_75 + 1.5*self._iqr
        min_idx = self._x >= lower_limit
        max_idx = self._x <= upper_limit
        out_idx = np.logical_not(np.logical_or(min_idx, max_idx))
        lower_whisker = np.min(self._x[min_idx])
        upper_whisker = np.max(self._x[max_idx])
        self.whiskers = np.array([lower_whisker, upper_whisker])
        self.outliers = self._x[out_idx]

    def _calc_params(self):
        self._calc_median()
        self._calc_percentiles()
        self._calc_iqr()


# def plot_boxplot3D(data1, data2, data3):
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
