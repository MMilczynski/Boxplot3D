import scipy
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle


class BoxplotParams:
    def __init__(self, x, whisker_type='IQR'):
        if not isinstance(x, np.ndarray):
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

    def calc_params(self):
        self._calc_median()
        self._calc_percentiles()
        self._calc_iqr()
        self._calc_min_max()
        self._calc_whiskers()


def plot_boxplot3D(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    bp_par_x = BoxplotParams(x)
    bp_par_y = BoxplotParams(y)
    bp_par_z = BoxplotParams(z)

    bp_par_x.calc_params()
    bp_par_y.calc_params()
    bp_par_z.calc_params()

    n_grid_pts = 3
    # z-data rectangles -> yp75 or yp25 fix
    z_rec_x = np.linspace(bp_par_x.perc_25, bp_par_x.perc_75, n_grid_pts)
    z_rec_z = np.linspace(bp_par_z.perc_25, bp_par_z.perc_75, n_grid_pts)
    z_rec_xx, z_rec_zz = np.meshgrid(z_rec_x, z_rec_z)

    z_rec_y_front = np.repeat(bp_par_y.perc_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.perc_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    ax.plot_surface(z_rec_xx, z_rec_y_front, z_rec_zz)
    ax.plot_surface(z_rec_xx, z_rec_y_back, z_rec_zz)

    plt.show()


def test_boxpot3D():
    x = np.random.randn(100)*5 + 10
    y = np.random.randn(100)*0.5 + 30
    z = np.random.randn(100)*3.5 + 20
    plot_boxplot3D(x, y, z)
