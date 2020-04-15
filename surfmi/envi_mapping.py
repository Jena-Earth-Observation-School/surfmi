

import numpy as np
from surfmi.envi_raster import *
import math
import matplotlib.pyplot as plt


def normalize(array):
    """
    Normalizes a two-dimensional numpy array based on the 2. percentile and 98. percentile.
    :param array: EnviRaster.ndarray[:,:,?]
    :return: numpy array
    """
    array_p2 = np.nanpercentile(array, 2)
    array_p98 = np.nanpercentile(array, 98)

    return (array - array_p2) / (array_p98 - array_p2)


def rgb(self, r, g, b):
    """
    Individual layers of an EnviRaster object can be selected using the RGB specifications. These layers are normalized
    by the function above. The return is a 3-layer EnviRaster object which allows the display of a RGB image using
    matplotlib's imshow.

    :param self: EnviRaster object
    :param r: Integer value defining the layer selection for the red channel
    :param g: Integer value defining the layer selection for the red channel
    :param b: Integer value defining the layer selection for the red channel
    :return: EnviRaster object
    """
    # Normalizing each layer and stack the results
    r_layer = normalize(self.ndarray[:, :, r])
    g_layer = normalize(self.ndarray[:, :, g])
    b_layer = normalize(self.ndarray[:, :, b])
    r_g_b = np.dstack((r_layer, g_layer, b_layer))

    # Extraction of header information
    header = self.header[self.header.array_id.isin([r, g, b])]

    return EnviRaster(header=header,
                      ndarray=r_g_b,
                      trans=self.trans,
                      proj=self.proj)


def greyscale(self, layer):
    """
    An Individual layer of an EnviRaster object can be selected. This layer gets normalized
    by the function above. The return is a EnviRaster object with one layer,
    which allows the display of an image in greyscale using matplotlib's imshow.

    :param self: EnviRaster object
    :param layer: Integer value defining the layer selection for the grayscaled image
    :return: EnviRaster object
    """
    # normalizing each layer and extraction of header information
    if len(self.ndarray.shape) == 2:
        array = normalize(self.ndarray)
    if len(self.ndarray.shape) == 3:
        array = normalize(self.ndarray[:, :, layer])
    header = self.header[self.header.array_id == layer]

    return EnviRaster(header=header,
                      ndarray=array,
                      trans=self.trans,
                      proj=self.proj)


def get_x_axislabels(self, nr_labels):
    """
    The function is a basis for labelling matplotlib's imshow x-axis (plt.xticks=) with coordinates.
    The required information is taken from the EnviRaster.trans instance.

    :param self: EnviRaster object
    :param nr_labels: Integer value defining the numbers of ticks at the x-axis
    :return: Position and labels as arrays
    """
    # Extracting the start-, end-coordinate & pixel size
    x_start = self.trans[0]
    x_p_sp = self.trans[1]
    x_end = x_start + self.ndarray.shape[1] * x_p_sp

    # Create all possible steps
    x = np.arange(x_start, x_end, x_p_sp)
    # Define each step size with nr_labels
    step_x = int(len(x) / nr_labels)+1
    # Get positions & labels
    x_positions = np.arange(0, len(x), step_x)
    x_labels = np.around(x[::step_x])

    return x_positions, x_labels


def get_y_axislabels(self, nr_labels):
    """
    The function is a basis for labelling matplotlib's imshow y-axis (plt.yticks=) with coordinates.
    The required information is taken from the EnviRaster.trans instance.

    :param self: EnviRaster object
    :param nr_labels: Integer value defining the numbers of ticks at the x-axis
    :return: Position and labels as arrays
    """
    # Extracting the start-, end-coordinate & pixel size
    y_start = self.trans[3]
    y_p_sp = self.trans[5]
    y_end = y_start + self.ndarray.shape[0] * y_p_sp

    # Create all possible steps
    y = np.arange(y_start, y_end, y_p_sp)
    # Define each step size with nr_labels
    step_y = int(len(y) / nr_labels) + 1
    # Get positions & labels
    y_positions = np.arange(0, len(y), step_y)
    y_labels = np.around(y[::step_y])

    return y_positions, y_labels


def mapping(figure_list, columns, nr_labels):
    """
    Function creates a matplotlib imshow plot with coordinates as axis-ticks.

    :param figure_list: List containing EnviRaster objects.
                        To get an optimal representation, the objects should be generated using rgb() or greyscale()
    :param columns: Integer indicating the numbers of columns in the plot
    :param nr_labels: Integer indicating the amount of ticks at each axis
    :return: matplotlib plot
    """
    # Define the layout by columns, rows & position
    row = math.ceil(len(figure_list) / columns)
    position = int(str(row) + str(columns) + "1")

    # Set a plot as example for each following one
    fig = plt.figure()
    ax1 = fig.add_subplot(position)
    # Define the labels and their position
    x_position, x_labels = get_x_axislabels(figure_list[0], nr_labels)
    y_position, y_labels = get_y_axislabels(figure_list[0], nr_labels)
    plt.xticks(x_position, x_labels)
    plt.yticks(y_position, y_labels)
    # Define the font size
    plt.rcParams.update({'font.size': 7})
    # Hide example plot
    plt.axis("off")

    # create a plot for each list element at position i
    for i, figure in enumerate(figure_list, position):
        ax = fig.add_subplot(i, sharex=ax1, sharey=ax1)
        # get layernames
        title = ""
        for bandname in figure.header.layernames:
            title = title + "\n" + bandname
        ax.set_title(title, color="blue")
        plt.imshow(figure.ndarray, "Greys", vmin=0, vmax=1)
