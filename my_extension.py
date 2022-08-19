import logging
import knime_extension as knext

from sklearn.metrics import DistanceMetric
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

@knext.node(name="Geo Distances", node_type=knext.NodeType.MANIPULATOR, icon_path="icons/my_node_icon.png", category="/")
@knext.input_table(name="Distances Table", description="Location pairs")
@knext.input_table(name="Coordinates Table", description="Lat/Long Coordinates of Cities")
@knext.output_table(name="Output Data", description="Distances in KM Attached to Second Input")
@knext.output_view("Scatter View", "Distances in a scatter plot")
class GeoDistances:
    """

    This node is able to compute distances between locations given latitude and longitude coordinates

    """

    dist_column = knext.StringParameter("Title of Distance Column", "Name for the new column reporting distances.", "Distance (Km)")
    loc_column = knext.ColumnParameter("Location Column", "Name of locations as reported in second input", port_index=1)
    lat_column = knext.ColumnParameter("Latitude Column", "Latitude column in decimal format", port_index=1)
    long_column = knext.ColumnParameter("Longitude Column", "Longitude column in decimal format", port_index=1)


    def configure(self, configure_context, input_schema_1, input_schema_2):
        return input_schema_1.append(knext.Column(knext.double(), name=self.dist_column))
 
    def execute(self, exec_context, input_1, input_2):

        exec_context.set_warning("This node was developed on Friday evening.")

        df_1_cities = input_1.to_pandas() 
        df_2_coords = input_2.to_pandas()

        distance_col_name = self.dist_column
        location_col_name = self.loc_column
        latitude_col_name = self.lat_column
        longitude_col_name = self.long_column

        if len(distance_col_name) > 100:
            raise ValueError("Too Long Column Name")

        df_2_coords[latitude_col_name] = np.radians(df_2_coords[latitude_col_name])
        df_2_coords[longitude_col_name] = np.radians(df_2_coords[longitude_col_name])

        array_with_city_names = df_2_coords[location_col_name].unique()

        dist = DistanceMetric.get_metric('haversine')
        earth_radius_km = 6373

        dist_array = dist.pairwise(df_2_coords[[latitude_col_name, longitude_col_name]].to_numpy()) * earth_radius_km
        distance_df = pd.DataFrame(dist_array, columns=array_with_city_names, index=array_with_city_names)

        from_city_col = df_1_cities.columns[0]
        to_city_col = df_1_cities.columns[1]

        dist_AB = []
        for couple_city in zip(df_1_cities[from_city_col], df_1_cities[to_city_col]):
            dist = distance_df.loc[couple_city[0], couple_city[1]]
            LOGGER.warning(f"The distance between {couple_city[0]} and {couple_city[1]} is {dist}")
            dist_AB.append(dist)
        df_1_cities[distance_col_name] = dist_AB

        cities = df_2_coords[location_col_name]

        x_line = df_1_cities[from_city_col]
        y_line = df_1_cities[to_city_col]

        fig, ax = plt.subplots()
        plt.title(distance_col_name)

        x_pixel = df_2_coords[longitude_col_name]
        y_pixel = df_2_coords[latitude_col_name]

        stdx = np.std(x_pixel) / 2
        stdy = np.std(y_pixel) / 2

        maxX = max(x_pixel + stdx)
        maxY = max(y_pixel + stdy)
        minX = min(x_pixel - stdx)
        minY = min(y_pixel - stdy)

        for i, txt in enumerate(df_2_coords[location_col_name].tolist()):
            ax.annotate(txt, (x_pixel[i]-stdx/2, y_pixel[i]-stdy/2))

        plt.xlim([minX,maxX])
        plt.ylim([minY,maxY])

        for i in range(len(x_line)):
            LOGGER.warning("Drawing: " + x_line[i] + " -> " + y_line[i])
            lats_longs_mini_df = df_2_coords.loc[(cities==x_line[i]) | (cities==y_line[i])]
            lats = lats_longs_mini_df[latitude_col_name].tolist()
            longs = lats_longs_mini_df[longitude_col_name].tolist()
            plt.plot(longs, lats, 'ro-')

            mid_point_x = (longs[0] + longs[1])/2
            mid_point_y = (lats[0] + lats[1])/2

            ax.annotate(str(dist_AB[i].round(2)) + " Km", (mid_point_x,mid_point_y))
        
        return knext.Table.from_pandas(df_1_cities), knext.view_matplotlib()