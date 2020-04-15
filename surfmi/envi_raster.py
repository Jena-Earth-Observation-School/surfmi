

import gdal
import numpy as np
import pandas as pd
import re


class EnviRaster:
    """
    This class simplifies the handling of Envi raster data. Especially for in- and export using gdal.
    In addition, arrays can be addressed and aggregated based on time or layername specifications.

    The class instances are:
    .header:
    - pandas time series dataframe listing "layernames", "date" and "array_id"
    .ndarray:
    - stores the raster data as one- or multidimensional numpy array
    .trans:
    - contains information about the georeference (most north-western point & pixel size)
    .proj:
    - contains information about the projection

    """

    def __init__(self, header, ndarray, trans, proj):
        self.header = header
        self.ndarray = ndarray
        self.trans = trans
        self.proj = proj

    # Getter Functions
    # get a layername from array_id
    def get_name_fnr(self, array_id):
        return self.header.layernames[array_id]

    # get array_id from date
    def get_nr_fdate(self, date):
        return self.header["array_id"][self.header["date"] == date][0]

    def get_nr_fname(self, layername):
        return self.header["array_id"][self.header["layernames"] == layername][0]

    # get array from layername
    def get_array_fname(self, layername):
        array_id = self.get_nr_fname(layername)
        return self.ndarray[:, :, array_id]

    # get array from array_id
    def get_array_fnr(self, array_id):
        return self.ndarray[:, :, array_id]

    
    def import_envi(input_file, na_value=-99):
        """
        Function imports Sentinel-1 Envi Data

        :param input_file: Path specification to envi file.
        :param na_value: Define NoData Value. Default is -99.
        :return: Object of class Envi_Raster
        """

        # Open and load the array
        input_img = gdal.Open(input_file)  # Import envi data as array
        layers = []  # Get list of arrays
        for i in range(1, input_img.RasterCount + 1):
            layers.append(input_img.GetRasterBand(i).ReadAsArray())

        # Stack the arrays and define NA value
        array_stack = np.dstack(layers)
        array_stack[array_stack == na_value] = np.nan

        # Get layernames from header file
        header = open(input_file + ".hdr")  # Import Envi header to get the layernames as list
        header = header.read()

        header = re.split("band names =", header)[1]
        # header = re.split("{|}", header)[1]
        header = re.split("[{}]", header)[1]
        header = header.replace("\n", "").replace(" ", "")
        header = re.split(",", header)
        header = pd.DataFrame({"layernames": header})

        # Create dataframe for aggregated data or percentiles
        if "agg" in header.layernames[0][41:44] or "per" in header.layernames[0][37:40]:
            header["start_date"] = pd.to_datetime(header.layernames.str[15:23], format="%Y%m%d")
            header["end_date"] = pd.to_datetime(header.layernames.str[24:32], format="%Y%m%d")
            # add date used for zonal statistics
            header["date"] = pd.to_datetime(header.layernames.str[15:23], format="%Y%m%d")
            header.index = header["date"]
            header["array_id"] = np.arange(len(header))

        # Create time-series df for Sentinel-1 or moisture data
        else:
            header["date"] = pd.to_datetime(header.layernames.str[12:20], format="%Y%m%d")
            header.index = header["date"]
            header["array_id"] = np.arange(len(header))

        return (EnviRaster(header=header,
                           ndarray=array_stack,
                           trans=input_img.GetGeoTransform(),
                           proj=input_img.GetProjection()))

    def export_envi(self, outfile, na_value=-99):
        """
        Function exports an Envi_Raster object.


        :param outfile: Path specification
        :return: Envi file in in float 32 format
        :param na_value: Define NoData Value. Default is -99.
        """

        # Export for one dimensional array
        if len(self.ndarray.shape) == 2:
            # define rows and columns
            [cols, rows] = self.ndarray.shape
            # Create file
            outdriver = gdal.GetDriverByName("ENVI")
            out_data = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)
            # Export Data
            out_data.GetRasterBand(1).WriteArray(self.ndarray)
            out_data.GetRasterBand(1).SetDescription(self.header["layernames"][0])
            out_data.GetRasterBand(1).SetNoDataValue(na_value)

        # Export for multidimensional arrays
        else:
            # define rows, columns and amount of layer
            [cols, rows, z] = self.ndarray.shape
            # Create file
            outdriver = gdal.GetDriverByName("ENVI")
            out_data = outdriver.Create(str(outfile), rows, cols, z, gdal.GDT_Float32)
            # Write the arrays to the file (Different Index of GetRasterbands and array index)
            i = 0
            while i < len(self.ndarray[1, 1, :]):
                out_data.GetRasterBand(i + 1).WriteArray(self.ndarray[:, :, i])
                out_data.GetRasterBand(i + 1).SetDescription(self.header["layernames"][i])
                out_data.GetRasterBand(i + 1).SetNoDataValue(na_value)
                i += 1

        # Write the geo-reference
        out_data.SetGeoTransform(self.trans)
        # Write the projection
        out_data.SetProjection(self.proj)
        # Close File
        out_data = None

        print("Exported!")

    def percentile(self, percentile, start_date, end_date):
        """
        Function calculates a percentile between two points in time of an Envi_Raster.

        The resulting band name consists of:
        "S1___IW_____VV_{YYYYmmdd}_{YYYYmmdd}_{ccc}_per{n}"

        {YYYYmmdd}   representing the start and end date for calibration
        {ccc}        number of scenes used for the calculation
        {n}          the percentile


        :param percentile: Integer between 0-100
        :param start_date: String containing a date in "YYYY-mm-dd" - format
        :param end_date: String containing a date in "YYYY-mm-dd" - format
        :return: Envi_Raster object containing the desired percentile
        """

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Get all S1-dates within the start-end-time span
        datesrange = []
        for d in self.header["date"]:
            if start_date <= d <= end_date:
                datesrange.append(d)

        # Get the array_id of first and last date within the time span
        idx_start = self.header["array_id"][self.header["date"] == datesrange[0]][0]
        idx_end = self.header["array_id"][self.header["date"] == datesrange[-1]][0]

        # Calculation of the desired percentile using the start/stop index & ignoring nan
        perc_array = np.nanpercentile(self.ndarray[:, :, idx_start:idx_end], percentile, axis=2)

        # create layername
        nr_of_scenes = str(len(datesrange)).zfill(3)
        layernames = "S1___IW_____VV_" \
                    + str(start_date)[0:10].replace("-", "") + "_" \
                    + str(end_date)[0:10].replace("-", "") + "_" \
                    + nr_of_scenes \
                    + "_per" + str(percentile)

        # create header
        header = pd.DataFrame({"layernames": [layernames], "start_date": [start_date], "end_date": [end_date]})
        header["array_id"] = np.arange(len(header))

        return (EnviRaster(header=header,
                           ndarray=perc_array,
                           trans=self.trans,
                           proj=self.proj))

    def moisture(self, dry_ref, wet_ref, layername_suffix="_moist"):
        """
        Function calculating the relative surface moisture algorithm based on Sentinel-1 Data of class Envi_Raster.

        Algorithm based on:
        URBAN, M., BERGER, C., MUDAU, T., HECKEL, K., TRUCKENBRODT, J., ONYANGO ODIPO, V., SMIT, I. & SCHMULLIUS, C.
        (2018): Surface Moisture and Vegetation Cover Analysis for Drought Monitoring in the Southern Kruger National
        Park Using Sentinel-1, Sentinel-2, and Landsat-8. – Remote Sensing 10, 9, 1482.

        :param dry_ref: Lower percentile of class Envi_Raster
        :param wet_ref: Upper percentile of class Envi_Raster
        :param layername_suffix: Suffix for each layername, default is "_moist"
        :return: Envi_Raster object
        """

        # Create empty Envi_Raster based on the S1-input

        s1_moist = EnviRaster(header=self.header.copy(),
                              ndarray=np.empty(self.ndarray.shape),
                              trans=self.trans,
                              proj=self.proj)

        #  Add suffix to the layernames
        s1_moist.header["layernames"] = s1_moist.header["layernames"].astype(str) + layername_suffix

        # Calculate the change detection algorithm
        for i in range(0, len(s1_moist.ndarray[1, 1, :])):
            s1_moist.ndarray[:, :, i] = ((self.ndarray[:, :, i] - dry_ref.ndarray) / (
                    wet_ref.ndarray - dry_ref.ndarray)) * 100
            # set values lower than 0 to 0
            s1_moist.ndarray[:, :, i][s1_moist.ndarray[:, :, i] < 0] = 0
            # set values higher than 100 to 100
            s1_moist.ndarray[:, :, i][s1_moist.ndarray[:, :, i] > 100] = 100

        return s1_moist

    def aggregation(self, agg):
        """
        Function aggregates the layers by month, quarter or year and calculates the median within each aggregation step

        The resulting band name consists of:
        "S1___IW_____VV_{YYYY-mm-dd}_{YYYY-mm-dd}_{ccc}_agg_{a}"

        {YYYY-mm-dd} representing the start and end date of the specific aggregation step
        {ccc}        number of scenes used in the aggregation
        {a}          the aggregation parameter ("M","Q" or "Y")


        :param agg: "M", "Q", or "Y" for monthly, quarterly or yearly aggregation
        :return: Envi_Raster object
        """

        # get sequences containing min & max array_id of each aggregation step
        min_idx = self.header.resample(agg).min()["array_id"].dropna()
        max_idx = self.header.resample(agg).max()["array_id"].dropna()
        # get sequences containing start & end date of each aggregation step using (MS,QS,YS) arguments
        agg_start = (self.header.resample(str(agg) + "S").min()["array_id"].dropna()).index
        agg_end = (self.header.resample(agg).min()["array_id"].dropna()).index
        # count scenes in each aggregation step
        count_scenes = self.header.resample(agg).count()["array_id"]
        count_scenes = count_scenes[count_scenes != 0]

        # Create empty array and dataframe
        agg_array = np.empty([self.ndarray.shape[0], self.ndarray.shape[1], len(min_idx)])
        layernames = []

        #  Calculate the median of each aggregation step
        for i in range(0, len(min_idx)):
            #  adressing sm_stack[:,:,235:235] results in NA values
            if int(min_idx[i]) == int(max_idx[i]):
                agg_array[:, :, i] = self.ndarray[:, :, int(min_idx[i])]
            else:
                agg_array[:, :, i] = np.nanmedian(self.ndarray[:, :, int(min_idx[i]):int(max_idx[i])], axis=2)

            # create a list with layernames
            start_date = agg_start[i]
            end_date = agg_end[i]
            nr_of_scenes = str(count_scenes[i]).zfill(3)
            layername = "S1___IW_____VV_" \
                       + str(start_date)[0:10].replace("-", "") + "_" \
                       + str(end_date)[0:10].replace("-", "") + "_" \
                       + nr_of_scenes + \
                       "_agg_" + agg

            layernames.append(layername)

        # create the header
        header = pd.DataFrame({"layernames": layernames, "start_date": agg_start, "end_date": agg_end})
        header["array_id"] = np.arange(len(header))

        return (EnviRaster(header=header,
                           ndarray=agg_array,
                           trans=self.trans,
                           proj=self.proj))
