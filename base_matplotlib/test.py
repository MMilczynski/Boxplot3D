"""
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Boxplot3D.base_matplotlib.surface import PercentileSurfaces
from Boxplot3D.base_matplotlib.surface import WhiskerSurfaces
from Boxplot3D.base_matplotlib.parameters import Params
from Boxplot3D.base_matplotlib.boxplot_3d import plot_x_medians
from Boxplot3D.base_matplotlib.boxplot_3d import plot_y_medians
from Boxplot3D.base_matplotlib.boxplot_3d import plot_z_medians


def _generate_test_data():

    x = np.random.randn(1000)*5 + 10
    y = np.random.randn(500)*1.5 + 30
    z = np.random.randn(2500)*4.5 + 20
    return x, y, z


def test_params():
    x, y, z = _generate_test_data()
    par_x = Params()
    par_y = Params()
    par_z = Params()
    par_x.calculate_params(x)
    par_y.calculate_params(y)
    par_z.calculate_params(z)
    return par_x, par_y, par_z


def test_surface():
    """
    """
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    x_color = (0.9, 0.0, 0.3)
    y_color = (0.6, 0.0, 0.3)
    z_color = (0.45, 0.0, 0.3)
    x, y, z = _generate_test_data()

    bp_par_x = Params()
    bp_par_y = Params()
    bp_par_z = Params()
    bp_par_x.calculate_params(x)
    bp_par_y.calculate_params(y)
    bp_par_z.calculate_params(z)
    percentiles = PercentileSurfaces(data=bp_par_z,
                                     width=bp_par_x,
                                     pos=bp_par_y,
                                     dimension='z')

    percentiles.build(axes)

    whiskers = WhiskerSurfaces(data=bp_par_z,
                               width=bp_par_x,
                               pos=bp_par_y,
                               dimension='z')
    whiskers.build(axes)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')


def test_boxplot_3D():
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    x_color = (0.9, 0.0, 0.3)
    y_color = (0.6, 0.0, 0.3)
    z_color = (0.45, 0.0, 0.3)
    x, y, z = _generate_test_data()

    bp_par_x = Params()
    bp_par_y = Params()
    bp_par_z = Params()
    bp_par_x.calculate_params(x)
    bp_par_y.calculate_params(y)
    bp_par_z.calculate_params(z)

    perc_z = PercentileSurfaces(data=bp_par_z,
                                width=bp_par_x,
                                pos=bp_par_y,
                                dimension='z')
    perc_z.build(axes)

    whis_z = WhiskerSurfaces(data=bp_par_z,
                             width=bp_par_x,
                             pos=bp_par_y,
                             dimension='z')
    whis_z.build(axes)

    perc_x = PercentileSurfaces(data=bp_par_x,
                                width=bp_par_y,
                                pos=bp_par_z,
                                dimension='x')
    perc_x.build(axes)

    whis_x = WhiskerSurfaces(data=bp_par_x,
                             width=bp_par_y,
                             pos=bp_par_z,
                             dimension='x')

    whis_x.build(axes)

    perc_y = PercentileSurfaces(data=bp_par_y,
                                width=bp_par_z,
                                pos=bp_par_x,
                                dimension='y')
    perc_y.build(axes)

    whis_y = WhiskerSurfaces(data=bp_par_y,
                             width=bp_par_z,
                             pos=bp_par_x,
                             dimension='y')
    whis_y.build(axes)

    plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    test_surface()
    plt.show()
    time.sleep(5.0)
    plt.close("all")
    test_boxplot_3D()
    plt.show()
