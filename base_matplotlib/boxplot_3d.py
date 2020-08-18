# -*- coding: utf-8 -*-
"""
    boxplot_3D.py
    -------------

    :copyright 2020 Matthias Milczynski
    :license ??
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Boxplot3D.base_matplotlib.surface import Surface
from Boxplot3D.base_matplotlib.parameters import Params

SCALING = 0.02
X_COLOR = (0.9, 0.0, 0.3)
Y_COLOR = (0.6, 0.0, 0.3)
Z_COLOR = (0.45, 0.0, 0.3)
STYLE = '-'
ALPHA = 0.75


def plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    """
    """
    offset = SCALING*bp_par_z.iqr
    z_rec_x = np.linspace(bp_par_x.percentile_low,
                          bp_par_x.percentile_high, n_grid_pts)
    z_rec_z = np.linspace(bp_par_z.median - offset,
                          bp_par_z.median + offset, n_grid_pts)
    z_rec_xx, z_rec_zz = np.meshgrid(z_rec_x, z_rec_z)

    z_rec_y_front = np.repeat(
        bp_par_y.percentile_high, np.power(n_grid_pts, 2))
    z_rec_y_front = z_rec_y_front.reshape(n_grid_pts, n_grid_pts)

    z_rec_y_back = np.repeat(bp_par_y.percentile_low, np.power(n_grid_pts, 2))
    z_rec_y_back = z_rec_y_back.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(z_rec_xx, z_rec_y_front, z_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(z_rec_xx, z_rec_y_back, z_rec_zz, color='k', zorder=0.5)


def plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    """
    """
    offset = SCALING*bp_par_z.iqr
    x_rec_y = np.linspace(bp_par_y.percentile_low,
                          bp_par_y.percentile_high, n_grid_pts)
    x_rec_x = np.linspace(bp_par_x.median - offset,
                          bp_par_x.median + offset, n_grid_pts)
    x_rec_xx, x_rec_yy = np.meshgrid(x_rec_x, x_rec_y)

    x_rec_z_top = np.repeat(bp_par_z.percentile_high, np.power(n_grid_pts, 2))
    x_rec_z_top = x_rec_z_top.reshape(n_grid_pts, n_grid_pts)

    x_rec_z_bottom = np.repeat(
        bp_par_z.percentile_low, np.power(n_grid_pts, 2))
    x_rec_z_bottom = x_rec_z_bottom.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_top, color='k', zorder=0.5)
    axes.plot_surface(x_rec_xx, x_rec_yy, x_rec_z_bottom,
                      color='k', zorder=0.5)


def plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes, n_grid_pts):
    """
    """
    offset = SCALING*bp_par_y.iqr
    y_rec_z = np.linspace(bp_par_z.percentile_low,
                          bp_par_z.percentile_high, n_grid_pts)
    y_rec_y = np.linspace(bp_par_y.median - offset,
                          bp_par_y.median + offset, n_grid_pts)
    y_rec_zz, y_rec_yy = np.meshgrid(y_rec_z, y_rec_y)

    y_rec_x_left = np.repeat(bp_par_x.percentile_low, np.power(n_grid_pts, 2))
    y_rec_x_left = y_rec_x_left.reshape(n_grid_pts, n_grid_pts)

    y_rec_x_right = np.repeat(
        bp_par_x.percentile_high, np.power(n_grid_pts, 2))
    y_rec_x_right = y_rec_x_right.reshape(n_grid_pts, n_grid_pts)

    axes.plot_surface(y_rec_x_right, y_rec_yy, y_rec_zz, color='k', zorder=0.5)
    axes.plot_surface(y_rec_x_left, y_rec_yy, y_rec_zz, color='k', zorder=0.5)


def plot_z_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.percentile_low + bp_par_x.iqr/2
    y_pos = bp_par_y.percentile_low + bp_par_y.iqr/2
    z_pos_1 = [bp_par_z.percentile_high, bp_par_z.whiskers[1]]
    z_pos_2 = [bp_par_z.percentile_low, bp_par_z.whiskers[0]]
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_1,
              linestyle=STYLE, alpha=0.9)
    axes.plot([x_pos, x_pos], [y_pos, y_pos], z_pos_2,
              linestyle=STYLE, alpha=0.9)


def plot_x_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    y_pos = bp_par_y.percentile_low + bp_par_y.iqr/2
    z_pos = bp_par_z.percentile_low + bp_par_z.iqr/2
    x_pos_1 = [bp_par_x.percentile_high, bp_par_x.whiskers[1]]
    x_pos_2 = [bp_par_x.percentile_low, bp_par_x.whiskers[0]]
    axes.plot(x_pos_1, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)
    axes.plot(x_pos_2, [y_pos, y_pos], [z_pos, z_pos],
              linestyle=STYLE, alpha=0.9)


def plot_y_whiskers(bp_par_x, bp_par_y, bp_par_z, axes):
    x_pos = bp_par_x.percentile_low + bp_par_x.iqr/2
    z_pos = bp_par_z.percentile_low + bp_par_z.iqr/2
    y_pos_1 = [bp_par_y.percentile_high, bp_par_y.whiskers[1]]
    y_pos_2 = [bp_par_y.percentile_low, bp_par_y.whiskers[0]]
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

    plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)


def test_boxplot_3D():
    x = np.random.randn(1000)*5 + 10
    y = np.random.randn(500)*1.5 + 30
    z = np.random.randn(2500)*4.5 + 20
    plot(x, y, z)


if __name__ == '__main__':
    test_boxplot_3D()
    plt.show()
