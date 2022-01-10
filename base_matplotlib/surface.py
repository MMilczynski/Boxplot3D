"""Implementations on the PercentileSurfaces and WhiskerSurfaces classes.

This module implements the following classes:
    - PercentileSurfaces
    - WhiskerSurfaces

Both classes generate the corresponding groups of surfaces using meshgrids.
They also implement the plotting capabilities.
"""
import numpy as np
import Boxplot3D.base_matplotlib.config as config


class PercentileSurfaces:
    """Generates a total of four surfaces (planes in 3D space) using meshgrids.

                             ▲    _________
                          height |\        | <-- replica upper percentile (secondary)
                             ▼   |_\_______|
                                  __\______
    replica lower percentile --> |   \____ |___
           (secondary)           |___|_____|   | <-- upper percentile (primary)
              ▲                   \  |_________|
    distance between replicas      \  _________ <-- space for median (median_half_width)
              ▼                     \|         | 
                                     |_________| <-- lower percentile (primary)
                                      ◄ width ►

    The four generated surfaces are:
        - Two surfaces representing lower and upper percentile of a
          "regular" boxplot with some space in between them for the median.
          This is the primary two-surface group.
        - A replica of the two-surface group shifted along the position axis.
          This is the secondary two-surface group.

    The height of the two primary surfaces combined is determined by boxplot parameters in
    the _data attribute. The _data attribute contains all the information of interest for
    the dimension of the primary two-surface group. The width of the primary surfaces is
    determined by boxplot parameters in the _width attribute which in fact will be the _data
    attribute of a neighboring surface. The distance between primary and secondary surface
    groups is determined by boxplot parameters in the _pos attribute.

    Refer to sketch above for some clarification.
    """

    def __init__(self,
                 data,
                 width,
                 pos,
                 dimension):
        """Initializes components required for the four surfaces. For details on the
        type of the _par attributes refer to the Params class in parameters.py.

        Args:
            data_par  : Boxplot parameters for data of interest along specified
                        dimension (see _dimension attribute). Specifies height of data
                        two-surface group based on lower and upper percentile.
            width_par : Boxplot parameters specifying the width of the two-surface
                        group.
            pos_par   : Boxplot parameters specifying the distance between actual two-surface
                        group and its shifted replicas.
            dimension : Either "x", "y" or "z".
        """
        self._data = data
        self._width = width
        self._pos = pos
        self._dimension = dimension

        # config params
        self._read_config_params()

        # mesh data
        self._lower_mesh = self._create_empty_meshgrid_set()
        self._upper_mesh = self._create_empty_meshgrid_set()
        self._primary_mesh = None
        self._secondary_mesh = None

        # 3D data
        self._x_data_1 = None
        self._x_data_2 = None
        self._y_data_1 = None
        self._y_data_2 = None
        self._z_data_1 = None
        self._z_data_2 = None

    def _read_config_params(self):
        """Read parameters from config.json into instance attributes.
        """
        self._median_half_width = config.CONFIG['Surface']['median_width_rel_iqr']*self._data.iqr
        self._n_grid_pts = config.CONFIG['Surface']['number_grid_points']
        self._alpha = config.CONFIG['Surface']['alpha']
        self._x_color = config.CONFIG['Surface']['x_color']
        self._y_color = config.CONFIG['Surface']['y_color']
        self._z_color = config.CONFIG['Surface']['z_color']


    def _create_empty_meshgrid_set(self):
        """Returns list of empty two-dimensional array based on number of grid-points
        specified.

        Returns:
            Two-element list of two-dimensional np-array for holding mesh-data.
        """
        meshgrid_set = []
        empty_mesh = np.zeros((self._n_grid_pts, self._n_grid_pts))
        meshgrid_set.append(empty_mesh)
        meshgrid_set.append(empty_mesh)
        return meshgrid_set

    def _gen_meshgrids(self):
        """Generate two meshgrids that are separated by the width of the median surface.
        """
        # 1. grid points for lower percentile data of data axis
        comp_1 = np.linspace(self._data.percentile_low,
                             self._data.median - self._median_half_width,
                             self._n_grid_pts)

        # 2. grid points for width of surface along axis spanning width. Note that
        # we're using .width here!
        comp_2 = np.linspace(self._width.percentile_low,
                             self._width.percentile_high,
                             self._n_grid_pts)

        # 3. grid points for upper percentile data of data axis
        comp_3 = np.linspace(self._data.median + self._median_half_width,
                             self._data.percentile_high,
                             self._n_grid_pts)

        # generate meshgrid for lower percentile surface that will extend
        # from the lowest percentile to the onset of the median surface
        self._lower_mesh[0], self._lower_mesh[1] = np.meshgrid(
            comp_1, comp_2)

        # generate grid for upper percentile surface that will extend
        # from the offset of the median surface to the highest percentile
        self._upper_mesh[0], self._upper_mesh[1] = np.meshgrid(
            comp_3, comp_2)

    def _replicate_meshgrids(self):
        """
        """
        # 1. grid points for primary meshgrid
        part_1 = np.repeat(self._pos.percentile_high,
                           np.power(self._n_grid_pts, 2))
        self._primary_mesh = part_1.reshape(self._n_grid_pts,
                                            self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        part_2 = np.repeat(self._pos.percentile_low,
                           np.power(self._n_grid_pts, 2))
        self._secondary_mesh = part_2.reshape(self._n_grid_pts,
                                              self._n_grid_pts)

    def _establish_x_order(self):
        """
        """
        self._x_data_1 = self._lower_mesh[0]
        self._x_data_2 = self._upper_mesh[0]
        self._y_data_1 = self._lower_mesh[1]
        self._y_data_2 = self._upper_mesh[1]
        self._z_data_1 = self._primary_mesh
        self._z_data_2 = self._secondary_mesh

    def _establish_y_order(self):
        """
        """
        self._x_data_1 = self._primary_mesh
        self._x_data_2 = self._secondary_mesh
        self._y_data_1 = self._lower_mesh[0]
        self._y_data_2 = self._upper_mesh[0]
        self._z_data_1 = self._lower_mesh[1]
        self._z_data_2 = self._upper_mesh[1]

    def _establish_z_order(self):
        """
        """
        self._x_data_1 = self._lower_mesh[1]
        self._x_data_2 = self._upper_mesh[1]
        self._y_data_1 = self._primary_mesh
        self._y_data_2 = self._secondary_mesh
        self._z_data_1 = self._lower_mesh[0]
        self._z_data_2 = self._upper_mesh[0]

    def build(self, axes):
        """
        """
        self._gen_meshgrids()
        self._replicate_meshgrids()
        getattr(self, '_establish_' + self._dimension + '_order')()
        getattr(self, '_plot_' + self._dimension + '_surface')(axes)

    def _plot_z_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._z_color)

    def _plot_x_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_2,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._x_color)

    def _plot_y_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._y_color)


class WhiskerSurfaces(PercentileSurfaces):
    """
    """
    def __init__(self, *arg, **kwarg):
        """
        """
        super().__init__(*arg, **kwarg)
        self._whisker_scaling = config.CONFIG['Surface']['whisker_scaling']

        self._low_mesh = self._create_empty_meshgrid_set()
        self._high_mesh = self._create_empty_meshgrid_set()
        self._low_mesh_rotated = self._create_empty_meshgrid_set()
        self._high_mesh_rotated = self._create_empty_meshgrid_set()

        self._x_data_1 = None
        self._x_data_2 = None
        self._y_data_1 = None
        self._y_data_2 = None
        self._z_data_1 = None
        self._z_data_2 = None

        self._x_data_1_rotated = None
        self._x_data_2_rotated = None
        self._y_data_1_rotated = None
        self._y_data_2_rotated = None
        self._z_data_1_rotated = None
        self._z_data_2_rotated = None

        self._primary_mesh = None
        self._secondary_mesh = None
        self._primary_mesh_rotated = None
        self._secondary_mesh_rotated = None

    def _gen_meshgrids(self):

        # I. Take care of data-facing meshgrid
        # 1. grid points for low-whisker surface along data axis
        comp_11 = np.linspace(self._data.percentile_low,
                              self._data.percentile_low - self._data.iqr,
                              self._n_grid_pts)
        low = self._width.median - self._width.iqr*self._whisker_scaling
        high = self._width.median + self._width.iqr*self._whisker_scaling

        comp_12 = np.linspace(low, high, self._n_grid_pts)

        comp_13 = np.linspace(self._data.percentile_high,
                              self._data.percentile_high + self._data.iqr,
                              self._n_grid_pts)

        # generate meshgrid for main low percentile whisker
        self._low_mesh[0], self._low_mesh[1] = np.meshgrid(
            comp_11, comp_12)

        # generate meshgrid for main high percentile whisker
        self._high_mesh[0], self._high_mesh[1] = np.meshgrid(
            comp_13, comp_12)

        # II. Take care of rotated meshgrid
        # 1. grid points for low-whisker surface along data axis
        comp_21 = np.linspace(self._data.percentile_low,
                              self._data.percentile_low - self._data.iqr,
                              self._n_grid_pts)
        low = self._pos.median - self._pos.iqr*self._whisker_scaling
        high = self._pos.median + self._pos.iqr*self._whisker_scaling

        comp_22 = np.linspace(low, high, self._n_grid_pts)

        comp_23 = np.linspace(self._data.percentile_high,
                              self._data.percentile_high + self._data.iqr,
                              self._n_grid_pts)

        # generate meshgrid for main low percentile whisker
        self._low_mesh_rotated[0], self._low_mesh_rotated[1] = np.meshgrid(
            comp_22, comp_21)

        # generate meshgrid for main high percentile whisker
        self._high_mesh_rotated[0], self._high_mesh_rotated[1] = np.meshgrid(
            comp_22, comp_23)

    def _replicate_meshgrids(self):
        """

        """
        # I. Replicate along position axis
        # 1. grid points for primary meshgrid
        high = self._pos.median + self._pos.iqr*self._whisker_scaling
        part_1 = np.repeat(high, np.power(self._n_grid_pts, 2))
        self._primary_mesh = part_1.reshape(self._n_grid_pts,
                                            self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        low = self._pos.median - self._pos.iqr*self._whisker_scaling
        part_2 = np.repeat(low, np.power(self._n_grid_pts, 2))
        self._secondary_mesh = part_2.reshape(self._n_grid_pts,
                                              self._n_grid_pts)

        # II. Replicate along width axis
        # 1. grid points for primary meshgrid
        high = self._width.median + self._width.iqr*self._whisker_scaling
        part_1 = np.repeat(high, np.power(self._n_grid_pts, 2))
        self._primary_mesh_rotated = part_1.reshape(self._n_grid_pts,
                                                    self._n_grid_pts)

        # 2. grid points for secondary meshgrid
        low = self._width.median - self._width.iqr*self._whisker_scaling
        part_2 = np.repeat(low, np.power(self._n_grid_pts, 2))
        self._secondary_mesh_rotated = part_2.reshape(self._n_grid_pts,
                                                      self._n_grid_pts)

    def _establish_z_order(self):
        """
        """
        self._x_data_1 = self._low_mesh[1]
        self._x_data_2 = self._high_mesh[1]
        self._y_data_1 = self._primary_mesh
        self._y_data_2 = self._secondary_mesh
        self._z_data_1 = self._low_mesh[0]
        self._z_data_2 = self._high_mesh[0]

        self._x_data_1_rotated = self._primary_mesh_rotated
        self._x_data_2_rotated = self._secondary_mesh_rotated
        self._y_data_1_rotated = self._low_mesh_rotated[0]
        self._y_data_2_rotated = self._high_mesh_rotated[0]
        self._z_data_1_rotated = self._low_mesh_rotated[1]
        self._z_data_2_rotated = self._high_mesh_rotated[1]

    def _establish_x_order(self):
        """
        """
        self._x_data_1 = self._low_mesh[0]
        self._x_data_2 = self._high_mesh[0]
        self._y_data_1 = self._low_mesh[1]
        self._y_data_2 = self._high_mesh[1]
        self._z_data_1 = self._primary_mesh
        self._z_data_2 = self._secondary_mesh

        self._x_data_1_rotated = self._low_mesh_rotated[1]
        self._x_data_2_rotated = self._high_mesh_rotated[1]
        self._y_data_1_rotated = self._primary_mesh_rotated
        self._y_data_2_rotated = self._secondary_mesh_rotated
        self._z_data_1_rotated = self._low_mesh_rotated[0]
        self._z_data_2_rotated = self._high_mesh_rotated[0]

    def _establish_y_order(self):
        """
        """
        self._x_data_1 = self._primary_mesh
        self._x_data_2 = self._secondary_mesh
        self._y_data_1 = self._low_mesh[0]
        self._y_data_2 = self._high_mesh[0]
        self._z_data_1 = self._low_mesh[1]
        self._z_data_2 = self._high_mesh[1]

        self._x_data_1_rotated = self._low_mesh_rotated[0]
        self._x_data_2_rotated = self._high_mesh_rotated[0]
        self._y_data_1_rotated = self._low_mesh_rotated[1]
        self._y_data_2_rotated = self._high_mesh_rotated[1]
        self._z_data_1_rotated = self._primary_mesh_rotated
        self._z_data_2_rotated = self._secondary_mesh_rotated

    def _plot_z_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_2,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_1,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._z_color)

        axes.plot_surface(self._x_data_1_rotated, self._y_data_1_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_1_rotated, self._y_data_2_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_1_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_2_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._z_color)

    def _plot_x_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_1,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_2,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._x_color)

        axes.plot_surface(self._x_data_1_rotated, self._y_data_1_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_1_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_1_rotated, self._y_data_2_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._z_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_2_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._z_color)

    def _plot_y_surface(self, axes):
        """
        """
        axes.plot_surface(self._x_data_1, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_1, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_2, self._y_data_1, self._z_data_1,
                          alpha=self._alpha, color=self._y_color)
        axes.plot_surface(self._x_data_2, self._y_data_2, self._z_data_2,
                          alpha=self._alpha, color=self._y_color)

        axes.plot_surface(self._x_data_1_rotated, self._y_data_1_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_2_rotated, self._z_data_1_rotated,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_1_rotated, self._y_data_1_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._x_color)
        axes.plot_surface(self._x_data_2_rotated, self._y_data_2_rotated, self._z_data_2_rotated,
                          alpha=self._alpha, color=self._x_color)
 
class WhiskerTicksSurfaces(WhiskerSurfaces):
    def __init__(self, *args, **kwargs):
        pass