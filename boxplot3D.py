import scipy
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle
from mayavi import mlab

median_thickness_scaling = 0.02
x_surface_color = (0.9, 0.0, 0.3)
y_surface_color = (0.6, 0.0, 0.3)
z_surface_color = (0.45, 0.0, 0.3)
whysker_style = '-'
alpha = 0.75


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
        self.iqr = None
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
        self.iqr = self.perc_75 - self.perc_25

    def _calc_min_max(self):
        self._min = np.min(self._x)
        self._max = np.max(self._x)

    def _calc_whiskers(self):
        lower_limit = self.perc_25 - 1.5*self.iqr
        upper_limit = self.perc_75 + 1.5*self.iqr
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


class AxisRectanglePair:
    def __init__(self, ax, bp_par_data, bp_par_width, bp_par_const):
        self._ax = ax
        self._bp_par_data = bp_par_data
        self._bp_par_width = bp_par_width
        self._bp_par_const = bp_par_const
        self._coords_xx = None
        self._coords_yy = None
        self._coords_front = None
        self._coords_back = None

    def calc_coordinates(self):
        pass


def plot_z_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # z-data rectangles
    iqr_offset = median_thickness_scaling*bp_par_z.iqr
    z_rec_x = np.linspace(bp_par_x.perc_25, bp_par_x.perc_75, n_grid_pts)
    z_rec_z_1 = np.linspace(
        bp_par_z.perc_25, bp_par_z.median - iqr_offset, n_grid_pts)
    z_rec_xx_1, z_rec_zz_1 = np.meshgrid(z_rec_x, z_rec_z_1)

    z_rec_z_2 = np.linspace(bp_par_z.median + iqr_offset,
                            bp_par_z.perc_75, n_grid_pts)
    z_rec_xx_2, z_rec_zz_2 = np.meshgrid(z_rec_x, z_rec_z_2)

    z_rec_y_front = np.repeat(bp_par_y.perc_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.perc_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(z_rec_xx_1, z_rec_y_front,
                      z_rec_zz_1, alpha=alpha, color=z_surface_color)
    axes.plot_surface(z_rec_xx_2, z_rec_y_front,
                      z_rec_zz_2, alpha=alpha, color=z_surface_color)
    axes.plot_surface(z_rec_xx_1, z_rec_y_back, z_rec_zz_1,
                      alpha=alpha, color=z_surface_color)
    axes.plot_surface(z_rec_xx_2, z_rec_y_back, z_rec_zz_2,
                      alpha=alpha, color=z_surface_color)


def plot_z_rectangles_mlab(bp_par_x, bp_par_y, bp_par_z, n_grid_pts):
    z_rec_x = np.linspace(bp_par_x.perc_25, bp_par_x.perc_75, n_grid_pts)
    z_rec_z = np.linspace(
        bp_par_z.perc_25, bp_par_z.perc_75, n_grid_pts)
    z_rec_xx, z_rec_zz = np.meshgrid(z_rec_x, z_rec_z)

    z_rec_y_front = np.repeat(bp_par_y.perc_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.perc_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    mlab.mesh(z_rec_xx, z_rec_y_front,
              z_rec_zz)
    mlab.mesh(z_rec_xx, z_rec_y_back, z_rec_zz)


def plot_x_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # x-data rectangles
    iqr_offset = median_thickness_scaling*bp_par_x.iqr
    x_rec_x_1 = np.linspace(
        bp_par_x.perc_25, bp_par_x.median - iqr_offset, n_grid_pts)
    x_rec_y = np.linspace(bp_par_y.perc_25, bp_par_y.perc_75, n_grid_pts)
    x_rec_xx_1, x_rec_yy_1 = np.meshgrid(x_rec_x_1, x_rec_y)

    x_rec_x_2 = np.linspace(
        bp_par_x.median + iqr_offset, bp_par_x.perc_75, n_grid_pts)
    x_rec_xx_2, x_rec_yy_2 = np.meshgrid(x_rec_x_2, x_rec_y)

    x_rec_z_top = np.repeat(bp_par_z.perc_75, np.power(n_grid_pts, 2))
    x_rec_z_top = x_rec_z_top.reshape(n_grid_pts, n_grid_pts)

    x_rec_z_bottom = np.repeat(bp_par_z.perc_25, np.power(n_grid_pts, 2))
    x_rec_z_bottom = x_rec_z_bottom.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(x_rec_xx_1, x_rec_yy_1, x_rec_z_top,
                      alpha=alpha, color=x_surface_color)
    axes.plot_surface(x_rec_xx_2, x_rec_yy_2, x_rec_z_top,
                      alpha=alpha, color=x_surface_color)
    axes.plot_surface(x_rec_xx_1, x_rec_yy_1, x_rec_z_bottom,
                      alpha=alpha, color=x_surface_color)
    axes.plot_surface(x_rec_xx_2, x_rec_yy_2, x_rec_z_bottom,
                      alpha=alpha, color=x_surface_color)


def plot_y_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # y-data rectangles
    iqr_offset = median_thickness_scaling*bp_par_y.iqr
    y_rec_y_1 = np.linspace(
        bp_par_y.perc_25, bp_par_y.median - iqr_offset, n_grid_pts)
    y_rec_z = np.linspace(bp_par_z.perc_25, bp_par_z.perc_75, n_grid_pts)
    y_rec_yy_1, y_rec_zz_1 = np.meshgrid(y_rec_y_1, y_rec_z)

    y_rec_y_2 = np.linspace(
        bp_par_y.median + iqr_offset, bp_par_y.perc_75, n_grid_pts)
    y_rec_yy_2, y_rec_zz_2 = np.meshgrid(y_rec_y_2, y_rec_z)

    y_rec_x_right = np.repeat(bp_par_x.perc_75, np.power(n_grid_pts, 2))
    y_rec_x_right = y_rec_x_right.reshape(n_grid_pts, n_grid_pts)

    y_rec_x_left = np.repeat(bp_par_x.perc_25, np.power(n_grid_pts, 2))
    y_rec_x_left = y_rec_x_left.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(y_rec_x_right, y_rec_yy_1, y_rec_zz_1,
                      alpha=alpha, color=y_surface_color)
    axes.plot_surface(y_rec_x_right, y_rec_yy_2, y_rec_zz_2,
                      alpha=alpha, color=y_surface_color)
    axes.plot_surface(y_rec_x_left, y_rec_yy_1, y_rec_zz_1,
                      alpha=alpha, color=y_surface_color)
    axes.plot_surface(y_rec_x_left, y_rec_yy_2, y_rec_zz_2,
                      alpha=alpha, color=y_surface_color)


def plot_z_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = median_thickness_scaling*bp_par_z.iqr
    z_rec_x = np.linspace(bp_par_x.perc_25, bp_par_x.perc_75, n_grid_pts)
    z_rec_z = np.linspace(bp_par_z.median - offset,
                          bp_par_z.median + offset, n_grid_pts)
    z_rec_xx, z_rec_zz = np.meshgrid(z_rec_x, z_rec_z)

    z_rec_y_front = np.repeat(bp_par_y.perc_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.perc_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(z_rec_xx, z_rec_y_front, z_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(z_rec_xx, z_rec_y_back, z_rec_zz, color='k', zorder=0.5)


def plot_x_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = median_thickness_scaling*bp_par_z.iqr
    x_rec_y = np.linspace(bp_par_y.perc_25, bp_par_y.perc_75, n_grid_pts)
    x_rec_x = np.linspace(bp_par_x.median - offset,
                          bp_par_x.median + offset, n_grid_pts)
    x_rec_xx, x_rec_yy = np.meshgrid(x_rec_x, x_rec_y)

    x_rec_z_top = np.repeat(bp_par_z.perc_75, np.power(n_grid_pts, 2))
    x_rec_z_top = x_rec_z_top.reshape(n_grid_pts, n_grid_pts)

    x_rec_z_bottom = np.repeat(bp_par_z.perc_25, np.power(n_grid_pts, 2))
    x_rec_z_bottom = x_rec_z_bottom.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_top, color='k', zorder=0.5)
    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_bottom,
                      color='k', zorder=0.5)


def plot_y_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = median_thickness_scaling*bp_par_y.iqr
    y_rec_z = np.linspace(bp_par_z.perc_25, bp_par_z.perc_75, n_grid_pts)
    y_rec_y = np.linspace(bp_par_y.median - offset,
                          bp_par_y.median + offset, n_grid_pts)
    y_rec_zz, y_rec_yy = np.meshgrid(y_rec_z, y_rec_y)

    y_rec_x_left = np.repeat(bp_par_x.perc_25, np.power(n_grid_pts, 2))
    y_rec_x_left = y_rec_x_left.reshape(n_grid_pts, n_grid_pts)

    y_rec_x_right = np.repeat(bp_par_x.perc_75, np.power(n_grid_pts, 2))
    y_rec_x_right = y_rec_x_right.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(y_rec_x_right, y_rec_yy, y_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(y_rec_x_left, y_rec_yy, y_rec_zz, color='k', zorder=0.5)


def plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, offset=0.1):
    color = 'k'
    linewidth = 2
    # z-medians
    axes.plot([bp_par_x.perc_25, bp_par_x.perc_75], [
        bp_par_y.perc_75 + offset, bp_par_y.perc_75 + offset], bp_par_z.median,
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.perc_25, bp_par_x.perc_75], [
        bp_par_y.perc_25 - offset, bp_par_y.perc_25 - offset], bp_par_z.median,
        color=color,
        linewidth=linewidth)


def plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes):
    color = 'k'
    linewidth = 2
    # x-medians
    axes.plot([bp_par_x.median, bp_par_x.median], [
        bp_par_y.perc_25, bp_par_y.perc_75], bp_par_z.perc_25,
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.median, bp_par_x.median], [
        bp_par_y.perc_25, bp_par_y.perc_75], bp_par_z.perc_75,
        color=color,
        linewidth=linewidth)


def plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes):
    color = 'k'
    linewidth = 2
    # x-medians
    axes.plot([bp_par_x.perc_25, bp_par_x.perc_25], [
        bp_par_y.median, bp_par_y.median], [bp_par_z.perc_25, bp_par_z.perc_75],
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.perc_75, bp_par_x.perc_75], [
        bp_par_y.median, bp_par_y.median], [bp_par_z.perc_25, bp_par_z.perc_75],
        color=color,
        linewidth=linewidth)


def plot_z_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.perc_25 + bp_par_x.iqr/2
    y_pos = bp_par_y.perc_25 + bp_par_y.iqr/2
    z_pos_1 = [bp_par_z.perc_75, bp_par_z.whiskers[1]]
    z_pos_2 = [bp_par_z.perc_25, bp_par_z.whiskers[0]]
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_1,
              linestyle=whysker_style, alpha=0.9)
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_2,
              linestyle=whysker_style, alpha=0.9)


def plot_x_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    y_pos = bp_par_y.perc_25 + bp_par_y.iqr/2
    z_pos = bp_par_z.perc_25 + bp_par_z.iqr/2
    x_pos_1 = [bp_par_x.perc_75, bp_par_x.whiskers[1]]
    x_pos_2 = [bp_par_x.perc_25, bp_par_x.whiskers[0]]
    axes.plot(x_pos_1, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=whysker_style, alpha=0.9)
    axes.plot(x_pos_2, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=whysker_style, alpha=0.9)


def plot_y_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.perc_25 + bp_par_x.iqr/2
    z_pos = bp_par_z.perc_25 + bp_par_z.iqr/2
    y_pos_1 = [bp_par_y.perc_75, bp_par_y.whiskers[1]]
    y_pos_2 = [bp_par_y.perc_25, bp_par_y.whiskers[0]]
    axes.plot([x_pos, x_pos], y_pos_1, [z_pos, z_pos],
              linestyle=whysker_style, alpha=0.9)
    axes.plot([x_pos, x_pos], y_pos_2, [z_pos, z_pos],
              linestyle=whysker_style, alpha=0.9)


def plot_boxplot3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.hold(True)
    bp_par_x = BoxplotParams(x)
    bp_par_y = BoxplotParams(y)
    bp_par_z = BoxplotParams(z)

    bp_par_x.calc_params()
    bp_par_y.calc_params()
    bp_par_z.calc_params()

    n_grid_pts = 3

    plot_z_whiskers(bp_par_x, bp_par_y, bp_par_z, ax)
    plot_x_whiskers(bp_par_x, bp_par_y, bp_par_z, ax)
    plot_y_whiskers(bp_par_x, bp_par_y, bp_par_z, ax)

    plot_z_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, alpha)
    plot_x_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, alpha)
    plot_y_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, alpha)

    plot_z_medians_2(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts)
    plot_x_medians_2(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts)
    plot_y_medians_2(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts)

    return (ax, bp_par_z)


def test_boxplot3D():
    x = np.random.randn(1000)*5 + 10
    y = np.random.randn(1000)*0.5 + 30
    z = np.random.randn(1000)*3.5 + 20
    return plot_boxplot3D(x, y, z)
