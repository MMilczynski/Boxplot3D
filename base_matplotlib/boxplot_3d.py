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
