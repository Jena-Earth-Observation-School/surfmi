

from surfmi.envi_raster import *
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import gdal
from osgeo import ogr
import numpy as np



def shape_to_array(shape_file, envi_raster):
    """
    Function converts a shapefile into a numpy array.
    As an intermediate step the shapefile is exported as tiff using gdal.
    The dimensions and properties of this tif file are based on an existing EnviRaster.

    Only if a pixel center is covered by the shapefile, the respective shapefile ID is assigned to the pixel.
    The NA value is set to 0.
    Finally a numpy array is created from the tif file.


    :param shape_file: Path specification to shapefile
    :param envi_raster: Object of class Envi_Raster
    :return: numpy array
    """

    # Load the shapefile
    shape = ogr.Open(shape_file)
    shape_lyr = shape.GetLayer()

    # Extract the Envi_Raster ndarray
    array_stack = envi_raster.ndarray

    # Create an empty rasterfile in tif format
    [cols, rows] = array_stack.shape[0:2]
    outdriver = gdal.GetDriverByName("GTiff")
    outfile = shape_file.split(".")[0] + ".tif"
    outdata = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(envi_raster.trans)
    outdata.SetProjection(envi_raster.proj)

    # Export Raster Band & Rasterize shape_lyr by id if the pixel centre is covered by the shapefile
    band = outdata.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(outdata, [1], shape_lyr, options=["ATTRIBUTE=id"])

    # Read the raster and return as array
    shape_array = outdata.GetRasterBand(1).ReadAsArray()
    shape_array[shape_array == 0] = np.nan
    return shape_array


def zonal_stats(s1_raster_file, shape_file):
    """
    The median and the standard deviation of the Envi raster are calculated using the areas predefined by the shapefile.
    The statistics are displayed in the resulting dataframe for each time point of the Envi raster.


    :param s1_raster_file: Path specification to envi file.

    :param shape_file:
    Path specification to shapefile. The shapefile must be in the same geographical CRS as the rasterfile.
    To examine different areas, the shapefile should contain different IDs

    :return:
    A dataframe is provided which lists the median and standard deviation of all points in time for each ID-area of
    the shapefile. The number in the column name identifies the shapefile ID and thus the chosen areas
    """

    # Load S1 product
    s1_raster = EnviRaster.import_envi(s1_raster_file)
    # Load mask
    array_mask = shape_to_array(shape_file, s1_raster)

    # Create dataframe
    s1_df = s1_raster.header[["layernames", "date"]].copy()
    # Get unique values of the mask array
    shape_classes = np.unique(array_mask[~np.isnan(array_mask)]).astype(int)

    # Get median and std values for each class
    for y in shape_classes:
        # Create a True False mask for specific class
        na_mask = array_mask == y

        median_list = []
        std_list = []
        for i in range(0, len(s1_raster.ndarray[1, 1, :])):
            # Calculate the median from the s1sm array indexed by the boolean mask
            median_list.append(np.nanmedian(s1_raster.ndarray[:, :, i][na_mask]))
            std_list.append(np.nanstd(s1_raster.ndarray[:, :, i][na_mask]))
        s1_df["median" + str(y)] = median_list
        s1_df["std" + str(y)] = std_list

    # print(s1_df.columns)
    return s1_df


def smooth(x, window_len, beta):
    """
    The function smooths a signal using a convolution based on the Kaiser window function from numpy.

    The convolution creates a weighted sum combination of the measured signal (x) and
    a window function (Kaiser window).

    The output signal is the median value of the weighted sum within a predetermined window width (window_len).
    The weighting is determined by the Kaiser window function and thus by the beta value

    Original code from https://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html


    :param x: Input signal as pandas series  (S1 median/std)
    :param window_len: Defining the window length as integer value
    :param beta: Defining the window function as integer value
    :return: Smoothed signal as pandas ndarray
    """

    # Extending the data at the beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, beta)
    y = np.convolve(w / w.sum(), s, mode="valid")

    # Plot some of the kaiser window functions
    # from matplotlib import pyplot
    # beta = [2,4,16,32,500]
    # for b in beta:
    #     w =np.kaiser(100,b)
    #     pyplot.plot(range(len(w)),w, label = str(b))

    return y[int(window_len / 2):len(y) - int(window_len / 2)]


def plotting(s1_df, s1_median, s1_color, legendname,
             slider_max, slider_interval, smoothing_window,
             title,
             html_out,

             precipitation_df=None,
             precipitation_selec=None,
             precipitation_col=None,

             soil_moisture_df=None,
             soil_moisture_selec=None,
             soil_moisture_col=None
             ):
    """
    This function enables the temporal representation of the zonal statistics dataframe.
    The median statistics can be selected and plotted according to their ID.

    The time series is smoothed by the smooth function. By increasing the smoothing_window a stronger
    smoothing can be achieved. The beta parameter can be set interactively in the plot.

    Additionally dataframes for precipitation and soil moisture can be plotted with separate y-axes.





    Sentinel-1 Data:
    :param s1_df:       Dataframe created by zonal_stats() function
    :param s1_median:   List containing a selection of median columns to be displayed
                            e.g. ["median1","median3"]
    :param s1_color:    List which contains the coloring of the individual median columns
                            e.g.["255,0,0", "0,0,255", ...]
    :param legendname:  List containing the names displayed in the legend


    Smoothing Paramters:
    :param slider_max:       Integer indicating the maximum of the beta value
    :param slider_interval:  Integer indicating the interval value
    :param smoothing_window: Integer indicating the smoothing window length

                                e.g. 11 is a good starting value

    Precipitation Data:
    :param precipitation_df:    Pandas dataframe with "date" & precipitation columns
    :param precipitation_selec: List containing a selection of columns to be displayed
                                    e.g. ["station1","station3"]
    :param precipitation_col:   List which contains the coloring of the individual columns
                                    e.g.["255,0,0", "0,0,255", ...]

    Soil Moisture Data
    :param soil_moisture_df:    Pandas dataframe with "date" & soil moisture columns
    :param soil_moisture_selec: List containing a selection of columns to be displayed
                                    e.g. ["lysimeter1","lysimeter3"]
    :param soil_moisture_col:   List which contains the coloring of the individual columns
                                    e.g.["255,0,0", "0,0,255", ...]

    :param title: Define the plot title
    :param html_out: Define the storage file for the html

    :return: Interactive Plotly Graph
    """

    #  Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces, for each s1_median & each smoothing step
    for i in range(0, len(s1_median)):

        # Delete NAs individually for each s1_median
        s1_df_na = s1_df[[s1_median[i], "std" + str(i + 1), "date"]].copy().dropna()

        for step in np.arange(0, slider_max, slider_interval):
            #  Add trace for upper std
            fig.add_trace(
                go.Scatter(
                    name='Upper Bound',
                    legendgroup=legendname[i],
                    showlegend=False,
                    x=s1_df_na["date"],
                    y=smooth(s1_df_na[s1_median[i]] + s1_df_na["std" + str(i + 1)] / 2, smoothing_window, step),
                    mode='lines',
                    line=dict(width=0),
                    fillcolor="rgba(" + s1_color[i] + ", 0.2)",
                    connectgaps=True,
                    hoverinfo="skip",
                    yaxis="y2"
                )
             )
            # Add trace for s1_median
            fig.add_trace(
                go.Scatter(
                    name=legendname[i],
                    legendgroup=legendname[i],
                    showlegend=True,
                    x=s1_df_na['date'],
                    y=smooth(s1_df_na[s1_median[i]], smoothing_window, step),
                    mode='lines',
                    line=dict(color="rgb(" + s1_color[i] + ")"),
                    fillcolor="rgba(" + s1_color[i] + ", 0.2)",
                    fill='tonexty',
                    connectgaps=True,
                    yaxis="y2"
                )
            )
            # Add trace for lower std
            fig.add_trace(
                go.Scatter(
                    name='Lower Bound',
                    legendgroup=legendname[i],
                    showlegend=False,
                    x=s1_df_na["date"],
                    y=smooth(s1_df_na[s1_median[i]] - s1_df_na["std" + str(i + 1)] / 2, smoothing_window, step),
                    mode='lines',
                    line=dict(width=0),
                    fillcolor="rgba(" + s1_color[i] + ", 0.2)",
                    fill='tonexty',
                    connectgaps=True,
                    hoverinfo="skip",
                    yaxis="y2"
                )
            )

    # All traces are stored in the fig.data tuple
    # For each step of the slider a True/False list is created,
    # specifying which trace should be displayed
    steps = []

    # Create a dict for each smoothing factor containing False entries
    for i in np.arange(0, slider_max, slider_interval):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)])

        # Switch (lower std (y), median (y+1), upper std(y+2)) for each slider step and each s1 data set to True
        # should use enumerate function
        index = list(range(0, slider_max, slider_interval)).index(i)
        for y in range(0, len(fig.data), int(len(fig.data) / len(s1_median))):
            step["args"][1][y + index * 3] = True
            step["args"][1][y + index * 3 + 1] = True
            step["args"][1][y + index * 3 + 2] = True
        steps.append(step)

        # Add True to each list end, in order to display the following precipitation and soil moisture traces
        # for any slider step
        if soil_moisture_selec or precipitation_selec:
            if soil_moisture_selec:
                nr_extend = len(soil_moisture_selec)
            if precipitation_selec:
                nr_extend = len(precipitation_selec)
            if soil_moisture_selec and precipitation_selec:
                nr_extend = len(soil_moisture_selec) + len(precipitation_selec)
            step["args"][1].extend([True] * nr_extend)

    #  Add traces for precipitation data
    if precipitation_selec:
        for i in range(0, len(precipitation_selec)):
            p_df_na = precipitation_df[[precipitation_selec[i], "date"]].copy().dropna()
            fig.add_trace(go.Bar(x=p_df_na.date,
                                 y=p_df_na[precipitation_selec[i]],
                                 name="Precipitation " + precipitation_selec[i],
                                 marker_color="rgb(" + precipitation_col[i] + ")",
                                 yaxis="y1"))

    # Add traces for soil_moisture data
    if soil_moisture_selec:
        for i in range(0, len(soil_moisture_selec)):
            sm_df_na = soil_moisture_df[[soil_moisture_selec[i], "date"]].copy().dropna()
            fig.add_trace(go.Scatter(x=sm_df_na.date,
                                     y=sm_df_na[soil_moisture_selec[i]],
                                     name="Soil moisture " + soil_moisture_selec[i],
                                     mode="lines",
                                     line=dict(color="rgb(" + soil_moisture_col[i] + ")"),
                                     line_shape="spline",
                                     line_smoothing=1.3,
                                     connectgaps=True,
                                     yaxis="y3"))

    # Layout
    # Foreground and background plotting is determined by the y-axis assignment.
    # Precipitation values are plotted with y1 in the background

    fig.update_layout(
        yaxis2=dict(
            title="Sentinel-1 Surface Moisture Index [%]",
            # range = [-25,125],
            overlaying="y",
            side="left",
        )
    )
    # If necessary add precipitation y-axis
    if precipitation_selec:
        fig.update_layout(
            yaxis=dict(
                title="Precipitation [mm]",
                range=[0, max(precipitation_df.select_dtypes(include=[np.number]).max())+10],
                side="right"
            )
        )

    # If necessary add soil moisture axis
    if soil_moisture_selec:
        fig.update_layout(
            yaxis3=dict(
                title="In Situ Soil Moisture [m^3/m^3]",
                range=[0, max(s1_df.select_dtypes(include=[np.number]).max()) / 2],
                # movable axis
                anchor="free",
                overlaying="y",
                side="right",
                position=1
            )
        )

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Smoothing Factor: "},
        pad={"t": 150},
        steps=steps
    )]

    fig.update_layout(
        autosize=True,
        legend_orientation="v",
        title_text=title,
        xaxis_rangeslider_visible=True,
        sliders=sliders,
        legend=dict(x=1.1, y=1))

    plot(fig, filename=html_out)


def smoothing_demonstration(s1_df, s1_median, s1_std,
                            slider_max, slider_interval, smoothing_window,
                            title, html_out):
    """
    The resulting plot allows the comparision between the smoothing results and the initial sequence.


    :param s1_df: Dataframe created by zonal_stats() function
    :param s1_median: String selecting one median column to be displayed
                            e.g. "median1"
    :param s1_std: String selecting one std column to be displayed
                        e.g. "std1"
    :param slider_max: Integer indicating the maximum of the beta value
    :param slider_interval: Integer indicating the interval value
    :param smoothing_window: Integer indicating the smoothing window length
                                e.g. 11 is a good starting value
    :param title: Define the plot title
    :param html_out: Define the storage file for the html
    :return:
    """

    #  Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Delete NAs individually for each s1_median
    s1_df_na = s1_df[[s1_median, s1_std, "date"]].copy().dropna()

    for step in np.arange(0, slider_max, slider_interval):
        #  Add trace for upper std
        fig.add_trace(
            go.Scatter(
                name='Upper Bound',
                legendgroup="Smoothed",
                showlegend=False,
                x=s1_df_na["date"],
                y=smooth(s1_df_na[s1_median] + s1_df_na[s1_std] / 2, smoothing_window, step),
                mode='lines',
                line=dict(width=0),
                fillcolor="rgba(255,0, 0, 0.2)",
                connectgaps=True,
                hoverinfo="skip",
                yaxis="y2"
            )
        )
        # Add trace for s1_median
        fig.add_trace(
            go.Scatter(
                name="Smoothed",
                legendgroup="Smoothed",
                showlegend=True,
                x=s1_df_na['date'],
                y=smooth(s1_df_na[s1_median], smoothing_window, step),
                mode='lines',
                line=dict(color="rgb(255,0, 0)"),
                fillcolor="rgba(255,0, 0, 0.2)",
                fill='tonexty',
                connectgaps=True,
                yaxis="y2"
            )
        )
        # Add trace for lower std
        fig.add_trace(
            go.Scatter(
                name='Lower Bound',
                legendgroup="Smoothed",
                showlegend=False,
                x=s1_df_na["date"],
                y=smooth(s1_df_na[s1_median] - s1_df_na[s1_std] / 2, smoothing_window, step),
                mode='lines',
                line=dict(width=0),
                fillcolor="rgba(255,0, 0, 0.2)",
                fill='tonexty',
                connectgaps=True,
                hoverinfo="skip",
                yaxis="y2"
            )
        )

    # All traces are stored in the fig.data tuple
    # For each step of the slider a True/False list is created,
    # specifying which trace should be displayed
    steps = []

    # Create lists for each smoothing factor containing False entries
    for i in np.arange(0, slider_max, slider_interval):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)])

        # Switch (lower std (y), median (y+1), upper std(y+2)) for each step and s1 data set to True
        index = list(range(0, slider_max, slider_interval)).index(i)
        for y in range(0, len(fig.data), int(len(fig.data))):
            step["args"][1][y + index * 3] = True
            step["args"][1][y + index * 3 + 1] = True
            step["args"][1][y + index * 3 + 2] = True
        steps.append(step)

        # Append 3 steps for the unsmoothed data
        step["args"][1].extend([True] * 3)

    # Add traces for unsmoothed upper std
    fig.add_trace(
        go.Scatter(
            name='Upper Bound',
            legendgroup="Unsmoothed",
            showlegend=False,
            x=s1_df_na["date"],
            y=s1_df_na[s1_median] + s1_df_na[s1_std] / 2,
            mode='lines',
            line=dict(width=0),
            fillcolor="rgba(0, 0, 255, 0.2)",
            connectgaps=True,
            hoverinfo="skip",
            yaxis="y1"
        )
    )

    # Add trace for unsmoothed s1_median
    fig.add_trace(
        go.Scatter(
            name="Unsmoothed",
            legendgroup="Unsmoothed",
            showlegend=True,
            x=s1_df_na['date'],
            y=s1_df_na[s1_median],
            mode='lines',
            line=dict(color="rgb(0, 0, 255)"),
            fillcolor="rgba(0, 0, 255, 0.2)",
            fill='tonexty',
            connectgaps=True,
            yaxis="y1"
        )
    )

    # Add trace for lower std
    fig.add_trace(
        go.Scatter(
            name='Lower Bound',
            legendgroup="Unsmoothed",
            showlegend=False,
            x=s1_df_na["date"],
            y=s1_df_na[s1_median] - s1_df_na[s1_std] / 2,
            mode='lines',
            line=dict(width=0),
            fillcolor="rgba(0, 0, 255, 0.2)",
            fill='tonexty',
            connectgaps=True,
            hoverinfo="skip",
            yaxis="y1"
        )
    )

    # Layout
    fig.update_layout(
        yaxis2=dict(
            title="Smoothed",
            overlaying="y",
            range=[min(s1_df_na[s1_median] - s1_df_na[s1_std] / 2) - 5,
                   max(s1_df_na[s1_median] + s1_df_na[s1_std] / 2) + 5],
            side="left"
        )
    )
    fig.update_layout(
        yaxis=dict(
            title="Unsmoothed",
            range=[min(s1_df_na[s1_median] - s1_df_na[s1_std] / 2) - 5,
                   max(s1_df_na[s1_median] + s1_df_na[s1_std] / 2) + 5],
            side="right"
        )
    )

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Smoothing Factor: "},
        pad={"t": 150},
        steps=steps
    )]

    fig.update_layout(
        autosize=True,
        legend_orientation="v",
        title_text=title,
        xaxis_rangeslider_visible=True,
        sliders=sliders,
        legend=dict(x=1.1, y=1))

    plot(fig, filename=html_out)
