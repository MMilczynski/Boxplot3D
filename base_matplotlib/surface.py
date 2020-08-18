import numpy as np
import matplotlib.pyplot as plt


class Surface:
    """Surface of boxplot 3D. 
    """

    def __init__(self,
                 data_par,
                 width_par,
                 pos_par,
                 order,
                 scaling,
                 alpha,
                 x_color,
                 y_color,
                 z_color):
        """
        """
        self._data_par = data_par
        self._width_par = width_par
        self._pos_par = pos_par
        self._iqr_offset = scaling*data_par.iqr
        self._n_grid_pts = 3
        self._order = order
        self._whisker_scaling = 0.05

        # refactor
        tmp_mat = np.zeros((self._n_grid_pts, self._n_grid_pts))

        self._low_percentile_mesh = [tmp_mat, tmp_mat]
        self._high_percentile_mesh = [tmp_mat, tmp_mat]
        self._low_whisker_mesh_main = [tmp_mat, tmp_mat]
        self._high_whisker_mesh_main = [tmp_mat, tmp_mat]
        self._low_whisker_mesh_rotated = [tmp_mat, tmp_mat]
        self._high_whisker_mesh_rotated = [tmp_mat, tmp_mat]

        self._x_percentile_data_1 = None
        self._x_percentile_data_2 = None
        self._y_percentile_data_1 = None
        self._y_percentile_data_2 = None
        self._z_percentile_data_1 = None
        self._z_percentile_data_2 = None

        self._x_whisker_main_data_1 = None
        self._x_whisker_main_data_2 = None
        self._y_whisker_main_data_1 = None
        self._y_whisker_main_data_2 = None
        self._z_whisker_main_data_1 = None
        self._z_whisker_main_data_2 = None

        self._x_whisker_rotated_data_1 = None
        self._x_whisker_rotated_data_2 = None
        self._y_whisker_rotated_data_1 = None
        self._y_whisker_rotated_data_2 = None
        self._z_whisker_rotated_data_1 = None
        self._z_whisker_rotated_data_2 = None

        self._primary_percentile_mesh = None
        self._secondary_percentile_mesh = None
        self._primary_whisker_mesh_main = None
        self._secondary_whisker_mesh_main = None
        self._primary_whisker_mesh_rotated = None
        self._secondary_whisker_mesh_rotated = None

        self._alpha = alpha
        self._x_color = x_color
        self._y_color = y_color
        self._z_color = z_color

    def _gen_whisker_meshgrids(self):

        # I. Take care of data-facing meshgrid
        # 1. grid points for low-whisker surface along data axis
        comp_11 = np.linspace(self._data_par.percentile_low,
                              self._data_par.percentile_low - self._data_par.iqr,
                              self._n_grid_pts)
        low = self._width_par.median - self._width_par.iqr*self._whisker_scaling
        high = self._width_par.median + self._width_par.iqr*self._whisker_scaling

        comp_12 = np.linspace(low, high, self._n_grid_pts)

        comp_13 = np.linspace(self._data_par.percentile_high,
                              self._data_par.percentile_high + self._data_par.iqr,
                              self._n_grid_pts)

        # generate meshgrid for main low percentile whisker
        self._low_whisker_mesh_main[0], self._low_whisker_mesh_main[1] = np.meshgrid(
            comp_11, comp_12)

        # generate meshgrid for main high percentile whisker
        self._high_whisker_mesh_main[0], self._high_whisker_mesh_main[1] = np.meshgrid(
            comp_13, comp_12)

        # II. Take care of rotated meshgrid
        # 1. grid points for low-whisker surface along data axis
        comp_21 = np.linspace(self._data_par.percentile_low,
                              self._data_par.percentile_low - self._data_par.iqr,
                              self._n_grid_pts)
        low = self._pos_par.median - self._pos_par.iqr*self._whisker_scaling
        high = self._pos_par.median + self._pos_par.iqr*self._whisker_scaling

        comp_22 = np.linspace(low, high, self._n_grid_pts)

        comp_23 = np.linspace(self._data_par.percentile_high,
                              self._data_par.percentile_high + self._data_par.iqr,
                              self._n_grid_pts)

        # generate meshgrid for main low percentile whisker
        self._low_whisker_mesh_rotated[0], self._low_whisker_mesh_rotated[1] = np.meshgrid(
            comp_22, comp_21)

        # generate meshgrid for main high percentile whisker
        self._high_whisker_mesh_rotated[0], self._high_whisker_mesh_rotated[1] = np.meshgrid(
            comp_22, comp_23)

    def _gen_percentile_meshgrids(self):
        """
        """
        # 1. grid points for low percentile data of data axis
        comp_1 = np.linspace(self._data_par.percentile_low,
                             self._data_par.median - self._iqr_offset,
                             self._n_grid_pts)

        # 2. grid points for width of surface along axis spaning width
        comp_2 = np.linspace(self._width_par.percentile_low,
                             self._width_par.percentile_high,
                             self._n_grid_pts)

        # 3. grid points for high percentile data of data axis
        comp_3 = np.linspace(self._data_par.median + self._iqr_offset,
                             self._data_par.percentile_high,
                             self._n_grid_pts)

        # generate meshgrid for low percentile surface
        self._low_percentile_mesh[0], self._low_percentile_mesh[1] = np.meshgrid(
            comp_1, comp_2)

        # generate grid for high percentile surface
        self._high_percentile_mesh[0], self._high_percentile_mesh[1] = np.meshgrid(
            comp_3, comp_2)

    def _replicate_percentile_meshgrids(self):
        """

        """
        # 1. grid points for primary meshgrid
        part_1 = np.repeat(self._pos_par.percentile_high,
                           np.power(self._n_grid_pts, 2))
        self._primary_percentile_mesh = part_1.reshape(self._n_grid_pts,
                                                       self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        part_2 = np.repeat(self._pos_par.percentile_low,
                           np.power(self._n_grid_pts, 2))
        self._secondary_percentile_mesh = part_2.reshape(self._n_grid_pts,
                                                         self._n_grid_pts)

    def _replicate_whisker_meshgrids(self):
        """

        """
        # I. Replicate along position axis
        # 1. grid points for primary meshgrid
        high = self._pos_par.median + self._pos_par.iqr*self._whisker_scaling
        part_1 = np.repeat(high, np.power(self._n_grid_pts, 2))
        self._primary_whisker_mesh_main = part_1.reshape(self._n_grid_pts,
                                                         self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        low = self._pos_par.median - self._pos_par.iqr*self._whisker_scaling
        part_2 = np.repeat(low, np.power(self._n_grid_pts, 2))
        self._secondary_whisker_mesh_main = part_2.reshape(self._n_grid_pts,
                                                           self._n_grid_pts)

        # II. Replicate along width axis
        # 1. grid points for primary meshgrid
        high = self._width_par.median + self._width_par.iqr*self._whisker_scaling
        part_1 = np.repeat(high, np.power(self._n_grid_pts, 2))
        self._primary_whisker_mesh_rotated = part_1.reshape(self._n_grid_pts,
                                                            self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        low = self._width_par.median - self._width_par.iqr*self._whisker_scaling
        part_2 = np.repeat(low, np.power(self._n_grid_pts, 2))
        self._secondary_whisker_mesh_rotated = part_2.reshape(self._n_grid_pts,
                                                              self._n_grid_pts)

    def _establish_x_percentile_order(self):
        """
        """
        self._x_percentile_data_1 = self._low_percentile_mesh[0]
        self._x_percentile_data_2 = self._high_percentile_mesh[0]
        self._y_percentile_data_1 = self._low_percentile_mesh[1]
        self._y_percentile_data_2 = self._high_percentile_mesh[1]
        self._z_percentile_data_1 = self._primary_percentile_mesh
        self._z_percentile_data_2 = self._secondary_percentile_mesh

    def _establish_y_percentile_order(self):
        """
        """
        self._x_percentile_data_1 = self._primary_percentile_mesh
        self._x_percentile_data_2 = self._secondary_percentile_mesh
        self._y_percentile_data_1 = self._low_percentile_mesh[0]
        self._y_percentile_data_2 = self._high_percentile_mesh[0]
        self._z_percentile_data_1 = self._low_percentile_mesh[1]
        self._z_percentile_data_2 = self._high_percentile_mesh[1]

    def _establish_z_percentile_order(self):
        """
        """
        self._x_percentile_data_1 = self._low_percentile_mesh[1]
        self._x_percentile_data_2 = self._high_percentile_mesh[1]
        self._y_percentile_data_1 = self._primary_percentile_mesh
        self._y_percentile_data_2 = self._secondary_percentile_mesh
        self._z_percentile_data_1 = self._low_percentile_mesh[0]
        self._z_percentile_data_2 = self._high_percentile_mesh[0]

    def _establish_z_whisker_order(self):
        """
        """
        self._x_whisker_main_data_1 = self._low_whisker_mesh_main[1]
        self._x_whisker_main_data_2 = self._high_whisker_mesh_main[1]
        self._y_whisker_main_data_1 = self._primary_whisker_mesh_main
        self._y_whisker_main_data_2 = self._secondary_whisker_mesh_main
        self._z_whisker_main_data_1 = self._low_whisker_mesh_main[0]
        self._z_whisker_main_data_2 = self._high_whisker_mesh_main[0]

        self._x_whisker_rotated_data_1 = self._primary_whisker_mesh_rotated
        self._x_whisker_rotated_data_2 = self._secondary_whisker_mesh_rotated
        self._y_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[0]
        self._y_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[0]
        self._z_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[1]
        self._z_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[1]

    def _establish_x_whisker_order(self):
        """
        """
        self._x_whisker_main_data_1 = self._low_whisker_mesh_main[0]
        self._x_whisker_main_data_2 = self._high_whisker_mesh_main[0]
        self._y_whisker_main_data_1 = self._low_whisker_mesh_main[1]
        self._y_whisker_main_data_2 = self._high_whisker_mesh_main[1]
        self._z_whisker_main_data_1 = self._primary_whisker_mesh_main
        self._z_whisker_main_data_2 = self._secondary_whisker_mesh_main

        self._x_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[1]
        self._x_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[1]
        self._y_whisker_rotated_data_1 = self._primary_whisker_mesh_rotated
        self._y_whisker_rotated_data_2 = self._secondary_whisker_mesh_rotated
        self._z_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[0]
        self._z_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[0]

    def _establish_y_whisker_order(self):
        """
        """
        self._x_whisker_main_data_1 = self._primary_whisker_mesh_main
        self._x_whisker_main_data_2 = self._secondary_whisker_mesh_main
        self._y_whisker_main_data_1 = self._low_whisker_mesh_main[0]
        self._y_whisker_main_data_2 = self._high_whisker_mesh_main[0]
        self._z_whisker_main_data_1 = self._low_whisker_mesh_main[1]
        self._z_whisker_main_data_2 = self._high_whisker_mesh_main[1]

        self._x_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[0]
        self._x_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[0]
        self._y_whisker_rotated_data_1 = self._low_whisker_mesh_rotated[1]
        self._y_whisker_rotated_data_2 = self._high_whisker_mesh_rotated[1]
        self._z_whisker_rotated_data_1 = self._primary_whisker_mesh_rotated
        self._z_whisker_rotated_data_2 = self._secondary_whisker_mesh_rotated

    def build_surface(self, axes):
        """
        """
        self._gen_percentile_meshgrids()
        self._replicate_percentile_meshgrids()
        getattr(self, '_establish_' + self._order + '_percentile_order')()
        getattr(self, '_plot_' + self._order + '_surface')(axes)

    def build_whiskers(self, axes):
        self._gen_whisker_meshgrids()
        self._replicate_whisker_meshgrids()
        getattr(self, '_establish_' + self._order + '_whisker_order')()
        getattr(self, '_plot_' + self._order + '_whiskers')(axes)

    def _plot_z_whiskers(self, axes):
        """
        """
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_1, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_1, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_2, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_2, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._z_color)

        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._z_color)

    def _plot_x_whiskers(self, axes):
        """
        """
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_1, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_2, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_1, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_2, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._x_color)

        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._z_color)

    def _plot_y_whiskers(self, axes):
        """
        """
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_1, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_whisker_main_data_1, self._y_whisker_main_data_2, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_1, self._z_whisker_main_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_whisker_main_data_2, self._y_whisker_main_data_2, self._z_whisker_main_data_2,
                          alpha=self._alpha, color=self._y_color)

        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_rotated_data_1, self._y_whisker_rotated_data_1, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_whisker_rotated_data_2, self._y_whisker_rotated_data_2, self._z_whisker_rotated_data_2,
                          alpha=self._alpha, color=self._x_color)

    def _plot_z_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_1, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_1, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_2, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_2, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._z_color)

    def _plot_x_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_1, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_2, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_1, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_2, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._x_color)

    def _plot_y_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_1, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_percentile_data_1, self._y_percentile_data_2, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_1, self._z_percentile_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_percentile_data_2, self._y_percentile_data_2, self._z_percentile_data_2,
                          alpha=self._alpha, color=self._y_color)
