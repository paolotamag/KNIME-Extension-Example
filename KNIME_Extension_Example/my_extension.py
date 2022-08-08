import logging
import knime_extension as knext

from sklearn.metrics import DistanceMetric
from math import radians
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


@knext.node(name="Geo Distances", node_type=knext.NodeType.LEARNER, icon_path="icons/my_node_icon.png", category="/")
@knext.input_table(name="Distances Table", description="Location pairs")
@knext.input_table(name="Coordinates Table", description="Lat/Long Coordinates of Cities")
@knext.output_table(name="Output Data", description="Distances in KM Attached to Second Input")

@knext.output_view("Scatter View", "Distances in a scatter plot")

class TemplateNode:
    """

    This node is able to compute distances between locations given latitude and longitude coordinates

    """



    another_param = knext.StringParameter("Title of Distance Column", "Name for the new column reporting distances.", "Distance (Km)")
    location_column = knext.ColumnParameter("Location Column", "Name of locations as reported in second input", port_index=1)
    lat_column = knext.ColumnParameter("Latitude Column", "Latitude column in decimal format", port_index=1)
    long_column = knext.ColumnParameter("Longitude Column", "Longitude column in decimal format", port_index=1)


    def configure(self, configure_context, input_schema_1, input_schema_2):
        return input_schema_1.append(knext.Column(knext.double(), name=self.another_param))

 
    def execute(self, exec_context, input_1, input_2):

        exec_context.set_warning("This node was developed on Friday evening.")

        input_1_pandas = input_1.to_pandas() 
        input_2_pandas = input_2.to_pandas()

        loc_col_name = self.location_column
        lat_col_name = self.lat_column
        long_col_name = self.long_column

        if len(self.another_param) > 100:
            raise ValueError("Too Long Column Name")

        

        input_2_pandas_new = input_2_pandas.copy()


        input_2_pandas_new[lat_col_name] = np.radians(input_2_pandas[lat_col_name])
        input_2_pandas_new[long_col_name] = np.radians(input_2_pandas[long_col_name])

        array_with_city_names = input_2_pandas_new[loc_col_name].unique()

        dist = DistanceMetric.get_metric('haversine')
        dist_array = dist.pairwise(input_2_pandas_new[[lat_col_name,long_col_name]].to_numpy())*6373
        distance_df = pd.DataFrame(dist_array, columns=array_with_city_names, index=array_with_city_names)

        from_city_col = input_1_pandas.columns[0]
        to_city_col = input_1_pandas.columns[1]

        dist_AB = []
        for couple_city in zip(input_1_pandas[from_city_col],input_1_pandas[to_city_col]):
            dist = distance_df.loc[couple_city[0],couple_city[1]]
            LOGGER.warning("The distance between "+ couple_city[0]+ " and "+ couple_city[1]+ " is "+ str(dist))
            dist_AB.append(dist)

        cities = input_2_pandas[loc_col_name]

        x_line = input_1_pandas[from_city_col]
        y_line = input_1_pandas[to_city_col]

        fig, ax = plt.subplots()
        plt.title(self.another_param)



        x_pixel = input_2_pandas[long_col_name]
        y_pixel = input_2_pandas[lat_col_name]

        stdx = np.std(x_pixel)/2
        stdy = np.std(y_pixel)/2

        maxX = max(x_pixel+stdx)
        maxY = max(y_pixel+stdy)
        minX = min(x_pixel-stdx)
        minY = min(y_pixel-stdy)


        for i, txt in enumerate(input_2_pandas[loc_col_name].tolist()):
            ax.annotate(txt, (x_pixel[i]-stdx/2, y_pixel[i]-stdy/2))



        plt.xlim([minX,maxX])
        plt.ylim([minY,maxY])

        for i in range(len(x_line)):
            LOGGER.warning("Drawing: " + x_line[i] + " -> " + y_line[i])
            lats_longs_mini_df = input_2_pandas.loc[(cities==x_line[i]) | (cities==y_line[i])]
            lats = lats_longs_mini_df[lat_col_name].tolist()
            longs = lats_longs_mini_df[long_col_name].tolist()
            plt.plot(longs, lats, 'ro-')

            mid_point_x = (longs[0] + longs[1])/2
            mid_point_y = (lats[0] + lats[1])/2

            ax.annotate(str(dist_AB[i].round(2)) + " Km", (mid_point_x,mid_point_y))

        input_1_pandas[self.another_param] = dist_AB
        
        return knext.Table.from_pandas(input_1_pandas), knext.view_matplotlib()