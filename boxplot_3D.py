# -*- coding: utf-8 -*-
"""
    boxplot_3D.py
    -------------

    :copyright 2019 Matthias Milczynski
    :license ??
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCALING = 0.02
X_COLOR = (0.9, 0.0, 0.3)
Y_COLOR = (0.6, 0.0, 0.3)
Z_COLOR = (0.45, 0.0, 0.3)
STYLE = '-'
ALPHA = 0.75


class Params:
    """Holding parameters of a 1-dimensional boxplot:
        - median
        - 25/75 percentiles
        - IQR i.e. inter-quantile range
        - whiskers
        - outliers
    """

    def __init__(self, whisker_type='IQR'):
        """
        :param whisker_type : either IQR or MinMax
        """
        if whisker_type not in ['IQR', 'MinMax']:
            raise Exception("Only whisker types 'IQR' and 'MinMax' allowed")
        self._whisker_type = whisker_type
        self._median = None
        self._percentile_25 = None
        self._percentile_75 = None
        self._iqr = None
        self._min = None
        self._max = None
        self._outliers = None
        self._whiskers = None
        self._upper_limit = None
        self._lower_limit = None
        self._data = None

    @property
    def percentile_25(self):
        return self._percentile_25

    @property
    def percentile_75(self):
        return self._percentile_75

    @property
    def iqr(self):
        return self._iqr

    @property
    def median(self):
        return self._median

    @property
    def whiskers(self):
        return self._whiskers

    def calculate_params(self, data):
        """Calculates all boxplot parameters.
        :param data: one-dimensional data in either list or np.array format
        """
        self._data = data
        if not isinstance(self._data, np.ndarray):
            self._data = np.array(self._data)
        self._calculate_median()
        self._calculate_percentiles()
        self._calculate_iqr()
        self._calculate_min_max()
        self._calculate_whiskers_iqr()

    def _calculate_median(self):
        """Calculates median.
        """
        self._median = np.median(self._data)

    def _calculate_percentiles(self):
        """Calculates 25 and 75 percentiles.
        """
        self._percentile_25 = np.percentile(self._data, 25)
        self._percentile_75 = np.percentile(self._data, 75)

    def _calculate_iqr(self):
        """Calculates inter-quantile range.
        """
        self._iqr = self._percentile_75 - self._percentile_25

    def _calculate_min_max(self):
        """Calculates min/max.
        """
        self._min = np.min(self._data)
        self._max = np.max(self._data)

    def _calculate_whiskers_iqr(self):
        """Calculates whiskers based on IQR. The upper extreme whisker is
        data-point with largest value within 75th percentile + 1.5*IQR.
        The lower extreme whisker data-point with lowest value within
        25th percentile - 1.5*IQR.
        """
        self._lower_limit = self._percentile_25 - 1.5*self._iqr
        self._upper_limit = self._percentile_75 + 1.5*self._iqr
        min_idx = self._data >= self._lower_limit
        max_idx = self._data <= self._upper_limit
        out_idx = np.logical_not(np.logical_or(min_idx, max_idx))
        lower_whisker = np.min(self._data[min_idx])
        upper_whisker = np.max(self._data[max_idx])
        self._whiskers = np.array([lower_whisker, upper_whisker])
        self._outliers = self._data[out_idx]


def plot_z_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # z-data rectangles
    iqr_offset = SCALING*bp_par_z.iqr
    z_rec_x = np.linspace(bp_par_x.percentile_25,
                          bp_par_x.percentile_75, n_grid_pts)
    z_rec_z_1 = np.linspace(
        bp_par_z.percentile_25, bp_par_z.median - iqr_offset, n_grid_pts)
    z_rec_xx_1, z_rec_zz_1 = np.meshgrid(z_rec_x, z_rec_z_1)

    z_rec_z_2 = np.linspace(bp_par_z.median + iqr_offset,
                            bp_par_z.percentile_75, n_grid_pts)
    z_rec_xx_2, z_rec_zz_2 = np.meshgrid(z_rec_x, z_rec_z_2)

    z_rec_y_front = np.repeat(bp_par_y.percentile_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.percentile_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(z_rec_xx_1, z_rec_y_front,
                      z_rec_zz_1, alpha=alpha, color=Z_COLOR)
    axes.plot_surface(z_rec_xx_2, z_rec_y_front,
                      z_rec_zz_2, alpha=alpha, color=Z_COLOR)
    axes.plot_surface(z_rec_xx_1, z_rec_y_back, z_rec_zz_1,
                      alpha=alpha, color=Z_COLOR)
    axes.plot_surface(z_rec_xx_2, z_rec_y_back, z_rec_zz_2,
                      alpha=alpha, color=Z_COLOR)


class Surface:
    """Surface of boxplot 3D. 
    """

    def __init__(self, data_par, width_par, pos_par, order):
        """
        """
        self._data_par = data_par
        self._width_par = width_par
        self._pos_par = pos_par
        self._iqr_offset = SCALING*data_par.iqr
        self._n_grid_pts = 3
        self._order = order

        tmp_mat = np.zeros((self._n_grid_pts, self._n_grid_pts))
        self._low_perc_surf = [tmp_mat, tmp_mat]
        self._high_perc_surf = [tmp_mat, tmp_mat]

        self._x_data_1 = None
        self._x_data_2 = None
        self._y_data_1 = None
        self._y_data_2 = None
        self._z_data_1 = None
        self._z_data_2 = None
        self._primary_surface = None
        self._secondary_surface = None

    def _gen_percentile_surface(self):
        """
        """
        comp_1 = np.linspace(self._data_par.percentile_25,
                             self._data_par.median - self._iqr_offset,
                             self._n_grid_pts)
        comp_2 = np.linspace(self._width_par.percentile_25,
                             self._width_par.percentile_75,
                             self._n_grid_pts)
        comp_3 = np.linspace(self._data_par.median + self._iqr_offset,
                             self._data_par.percentile_75, self._n_grid_pts)

        self._low_perc_surf[0], self._low_perc_surf[1] = np.meshgrid(
            comp_1, comp_2)
        self._high_perc_surf[0], self._high_perc_surf[1] = np.meshgrid(
            comp_3, comp_2)

    def _multiply_percentile_surfaces(self):
        """
        """
        # primary surface
        part_1 = np.repeat(self._pos_par.percentile_75,
                           np.power(self._n_grid_pts, 2))
        self._primary_surface = part_1.reshape(
            self._n_grid_pts, self._n_grid_pts)
        # secondary surface
        part_2 = np.repeat(self._pos_par.percentile_25,
                           np.power(self._n_grid_pts, 2))
        self._secondary_surface = part_2.reshape(
            self._n_grid_pts, self._n_grid_pts)

    def _establish_x_order(self):
        """
        """
        self._x_data_1 = self._low_perc_surf[0]
        self._x_data_2 = self._high_perc_surf[0]
        self._y_data_1 = self._low_perc_surf[1]
        self._y_data_2 = self._high_perc_surf[1]
        self._z_data_1 = self._primary_surface
        self._z_data_2 = self._secondary_surface

    def _establish_y_order(self):
        """
        """
        self._x_data_1 = self._primary_surface
        self._x_data_2 = self._secondary_surface
        self._y_data_1 = self._low_perc_surf[0]
        self._y_data_2 = self._high_perc_surf[0]
        self._z_data_1 = self._low_perc_surf[1]
        self._z_data_2 = self._high_perc_surf[1]

    def _establish_z_order(self):
        """
        """
        self._x_data_1 = self._low_perc_surf[1]
        self._x_data_2 = self._high_perc_surf[1]
        self._y_data_1 = self._primary_surface
        self._y_data_2 = self._secondary_surface
        self._z_data_1 = self._low_perc_surf[0]
        self._z_data_2 = self._high_perc_surf[0]

    def build_surface(self, axes):
        """
        """
        self._gen_percentile_surface()
        self._multiply_percentile_surfaces()
        getattr(self, '_establish_' + self._order + '_order')()
        getattr(self, '_plot_' + self._order + '_surface')(axes)

    def _plot_z_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)

    def _plot_x_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)

    def _plot_y_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_1,
                          alpha=ALPHA, color=X_COLOR)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=ALPHA, color=X_COLOR)


def plot_x_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # x-data rectangles
    iqr_offset = SCALING*bp_par_x.iqr
    x_rec_x_1 = np.linspace(
        bp_par_x.percentile_25, bp_par_x.median - iqr_offset, n_grid_pts)
    x_rec_y = np.linspace(bp_par_y.percentile_25,
                          bp_par_y.percentile_75, n_grid_pts)
    x_rec_xx_1, x_rec_yy_1 = np.meshgrid(x_rec_x_1, x_rec_y)

    x_rec_x_2 = np.linspace(
        bp_par_x.median + iqr_offset, bp_par_x.percentile_75, n_grid_pts)
    x_rec_xx_2, x_rec_yy_2 = np.meshgrid(x_rec_x_2, x_rec_y)

    x_rec_z_top = np.repeat(bp_par_z.percentile_75, np.power(n_grid_pts, 2))
    x_rec_z_top = x_rec_z_top.reshape(n_grid_pts, n_grid_pts)

    x_rec_z_bottom = np.repeat(bp_par_z.percentile_25, np.power(n_grid_pts, 2))
    x_rec_z_bottom = x_rec_z_bottom.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(x_rec_xx_1, x_rec_yy_1, x_rec_z_top,
                      alpha=alpha, color=X_COLOR)
    axes.plot_surface(x_rec_xx_2, x_rec_yy_2, x_rec_z_top,
                      alpha=alpha, color=X_COLOR)
    axes.plot_surface(x_rec_xx_1, x_rec_yy_1, x_rec_z_bottom,
                      alpha=alpha, color=X_COLOR)
    axes.plot_surface(x_rec_xx_2, x_rec_yy_2, x_rec_z_bottom,
                      alpha=alpha, color=X_COLOR)


def plot_y_rectangles(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts, alpha):
    # y-data rectangles
    iqr_offset = SCALING*bp_par_y.iqr
    y_rec_y_1 = np.linspace(
        bp_par_y.percentile_25, bp_par_y.median - iqr_offset, n_grid_pts)
    y_rec_z = np.linspace(bp_par_z.percentile_25,
                          bp_par_z.percentile_75, n_grid_pts)
    y_rec_yy_1, y_rec_zz_1 = np.meshgrid(y_rec_y_1, y_rec_z)

    y_rec_y_2 = np.linspace(
        bp_par_y.median + iqr_offset, bp_par_y.percentile_75, n_grid_pts)
    y_rec_yy_2, y_rec_zz_2 = np.meshgrid(y_rec_y_2, y_rec_z)

    y_rec_x_right = np.repeat(bp_par_x.percentile_75, np.power(n_grid_pts, 2))
    y_rec_x_right = y_rec_x_right.reshape(n_grid_pts, n_grid_pts)

    y_rec_x_left = np.repeat(bp_par_x.percentile_25, np.power(n_grid_pts, 2))
    y_rec_x_left = y_rec_x_left.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(y_rec_x_right, y_rec_yy_1, y_rec_zz_1,
                      alpha=alpha, color=Y_COLOR)
    axes.plot_surface(y_rec_x_right, y_rec_yy_2, y_rec_zz_2,
                      alpha=alpha, color=Y_COLOR)
    axes.plot_surface(y_rec_x_left, y_rec_yy_1, y_rec_zz_1,
                      alpha=alpha, color=Y_COLOR)
    axes.plot_surface(y_rec_x_left, y_rec_yy_2, y_rec_zz_2,
                      alpha=alpha, color=Y_COLOR)


def plot_z_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = SCALING*bp_par_z.iqr
    z_rec_x = np.linspace(bp_par_x.percentile_25,
                          bp_par_x.percentile_75, n_grid_pts)
    z_rec_z = np.linspace(bp_par_z.median - offset,
                          bp_par_z.median + offset, n_grid_pts)
    z_rec_xx, z_rec_zz = np.meshgrid(z_rec_x, z_rec_z)

    z_rec_y_front = np.repeat(bp_par_y.percentile_75, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.percentile_25, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(z_rec_xx, z_rec_y_front, z_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(z_rec_xx, z_rec_y_back, z_rec_zz, color='k', zorder=0.5)


def plot_x_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = SCALING*bp_par_z.iqr
    x_rec_y = np.linspace(bp_par_y.percentile_25,
                          bp_par_y.percentile_75, n_grid_pts)
    x_rec_x = np.linspace(bp_par_x.median - offset,
                          bp_par_x.median + offset, n_grid_pts)
    x_rec_xx, x_rec_yy = np.meshgrid(x_rec_x, x_rec_y)

    x_rec_z_top = np.repeat(bp_par_z.percentile_75, np.power(n_grid_pts, 2))
    x_rec_z_top = x_rec_z_top.reshape(n_grid_pts, n_grid_pts)

    x_rec_z_bottom = np.repeat(bp_par_z.percentile_25, np.power(n_grid_pts, 2))
    x_rec_z_bottom = x_rec_z_bottom.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_top, color='k', zorder=0.5)
    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_bottom,
                      color='k', zorder=0.5)


def plot_y_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    offset = SCALING*bp_par_y.iqr
    y_rec_z = np.linspace(bp_par_z.percentile_25,
                          bp_par_z.percentile_75, n_grid_pts)
    y_rec_y = np.linspace(bp_par_y.median - offset,
                          bp_par_y.median + offset, n_grid_pts)
    y_rec_zz, y_rec_yy = np.meshgrid(y_rec_z, y_rec_y)

    y_rec_x_left = np.repeat(bp_par_x.percentile_25, np.power(n_grid_pts, 2))
    y_rec_x_left = y_rec_x_left.reshape(n_grid_pts, n_grid_pts)

    y_rec_x_right = np.repeat(bp_par_x.percentile_75, np.power(n_grid_pts, 2))
    y_rec_x_right = y_rec_x_right.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(y_rec_x_right, y_rec_yy, y_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(y_rec_x_left, y_rec_yy, y_rec_zz, color='k', zorder=0.5)


def plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, offset=0.1):
    color = 'k'
    linewidth = 2
    # z-medians
    axes.plot([bp_par_x.percentile_25, bp_par_x.percentile_75], [
        bp_par_y.percentile_75 + offset, bp_par_y.percentile_75 + offset], bp_par_z.median,
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.percentile_25, bp_par_x.percentile_75], [
        bp_par_y.percentile_25 - offset, bp_par_y.percentile_25 - offset], bp_par_z.median,
        color=color,
        linewidth=linewidth)


def plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes):
    color = 'k'
    linewidth = 2
    # x-medians
    axes.plot([bp_par_x.median, bp_par_x.median], [
        bp_par_y.percentile_25, bp_par_y.percentile_75], bp_par_z.percentile_25,
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.median, bp_par_x.median], [
        bp_par_y.percentile_25, bp_par_y.percentile_75], bp_par_z.percentile_75,
        color=color,
        linewidth=linewidth)


def plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes):
    color = 'k'
    linewidth = 2
    # x-medians
    axes.plot([bp_par_x.percentile_25, bp_par_x.percentile_25], [
        bp_par_y.median, bp_par_y.median], [bp_par_z.percentile_25, bp_par_z.percentile_75],
        color=color,
        linewidth=linewidth)
    axes.plot([bp_par_x.percentile_75, bp_par_x.percentile_75], [
        bp_par_y.median, bp_par_y.median], [bp_par_z.percentile_25, bp_par_z.percentile_75],
        color=color,
        linewidth=linewidth)


def plot_z_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.percentile_25 + bp_par_x.iqr/2
    y_pos = bp_par_y.percentile_25 + bp_par_y.iqr/2
    z_pos_1 = [bp_par_z.percentile_75, bp_par_z.whiskers[1]]
    z_pos_2 = [bp_par_z.percentile_25, bp_par_z.whiskers[0]]
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_1,
              linestyle=STYLE, alpha=0.9)
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_2,
              linestyle=STYLE, alpha=0.9)


def plot_x_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    y_pos = bp_par_y.percentile_25 + bp_par_y.iqr/2
    z_pos = bp_par_z.percentile_25 + bp_par_z.iqr/2
    x_pos_1 = [bp_par_x.percentile_75, bp_par_x.whiskers[1]]
    x_pos_2 = [bp_par_x.percentile_25, bp_par_x.whiskers[0]]
    axes.plot(x_pos_1, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)
    axes.plot(x_pos_2, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)


def plot_y_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.percentile_25 + bp_par_x.iqr/2
    z_pos = bp_par_z.percentile_25 + bp_par_z.iqr/2
    y_pos_1 = [bp_par_y.percentile_75, bp_par_y.whiskers[1]]
    y_pos_2 = [bp_par_y.percentile_25, bp_par_y.whiskers[0]]
    axes.plot([x_pos, x_pos], y_pos_1, [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)
    axes.plot([x_pos, x_pos], y_pos_2, [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)


def plot(x, y, z):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    # ax.hold(True)
    bp_par_x = Params()
    bp_par_y = Params()
    bp_par_z = Params()

    bp_par_x.calculate_params(x)
    bp_par_y.calculate_params(y)
    bp_par_z.calculate_params(z)

    plot_z_whiskers(bp_par_x, bp_par_y, bp_par_z, axes)
    plot_x_whiskers(bp_par_x, bp_par_y, bp_par_z, axes)
    plot_y_whiskers(bp_par_x, bp_par_y, bp_par_z, axes)

    z_surf = Surface(data_par=bp_par_z,
                     width_par=bp_par_x,
                     pos_par=bp_par_y, order='z')
    z_surf.build_surface(axes)
    x_surf = Surface(data_par=bp_par_x,
                     width_par=bp_par_y,
                     pos_par=bp_par_z, order='x')
    x_surf.build_surface(axes)
    y_surf = Surface(data_par=bp_par_y,
                     width_par=bp_par_z,
                     pos_par=bp_par_x, order='y')
    y_surf.build_surface(axes)

    # plot_z_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, ALPHA)
    # plot_x_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, ALPHA)
    # plot_y_rectangles(bp_par_x, bp_par_y, bp_par_z, ax, n_grid_pts, ALPHA)

    plot_z_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_x_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_y_medians_2(bp_par_x, bp_par_y, bp_par_z, axes, 3)


def test_boxplot_3D():
    x = np.random.randn(1000)*5 + 10
    y = np.random.randn(1000)*0.5 + 30
    z = np.random.randn(1000)*3.5 + 20
    plot(x, y, z)


if __name__ == '__main__':
    test_boxplot_3D()
    plt.show()
