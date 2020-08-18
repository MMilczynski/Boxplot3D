import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Boxplot3D.base_matplotlib.surface import Surface
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
    surf = Surface(data_par=bp_par_z,
                   width_par=bp_par_x,
                   pos_par=bp_par_y,
                   order='z',
                   alpha=0.9,
                   x_color=x_color,
                   y_color=y_color,
                   z_color=z_color)

    surf.build_surface(axes)
    surf.build_whiskers(axes)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')


def test_boxplot3D():
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

    surf_z = Surface(data_par=bp_par_z,
                     width_par=bp_par_x,
                     pos_par=bp_par_y,
                     order='z',
                     alpha=0.9,
                     x_color=x_color,
                     y_color=y_color,
                     z_color=z_color)
    surf_z.build_surface(axes)
    surf_z.build_whiskers(axes)

    surf_x = Surface(data_par=bp_par_x,
                     width_par=bp_par_y,
                     pos_par=bp_par_z,
                     order='x',
                     alpha=0.9,
                     x_color=x_color,
                     y_color=y_color,
                     z_color=z_color)
    surf_x.build_surface(axes)
    surf_x.build_whiskers(axes)

    surf_y = Surface(data_par=bp_par_y,
                     width_par=bp_par_z,
                     pos_par=bp_par_x,
                     order='y',
                     alpha=0.9,
                     x_color=x_color,
                     y_color=y_color,
                     z_color=z_color)
    surf_y.build_surface(axes)
    surf_y.build_whiskers(axes)

    surf_y.build_surface(axes)
    surf_y.build_whiskers(axes)

    plot_z_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_x_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)
    plot_y_medians(bp_par_x, bp_par_y, bp_par_z, axes, 3)

    return surf_x, surf_y, surf_z


if __name__ == '__main__':
    test_surface()
    plt.show()
