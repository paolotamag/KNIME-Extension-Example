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

    location_column = knext.ColumnParameter("Location Column", "Name of locations as reported in second input", port_index=1)
    lat_column = knext.ColumnParameter("Latitude Column", "Latitude column in decimal format", port_index=1)
    long_column = knext.ColumnParameter("Longitude Column", "Longitude column in decimal format", port_index=1)
    output_column_name = knext.StringParameter("Title of Output Column", "Name for the new column reporting distances.", "Distance (Km)")


    def configure(self, configure_context, input_schema_1, input_schema_2):
        # Check that the parameters can be used
        if self.location_column is None or self.lat_column is None or self.long_column is None:
            raise knext.InvalidParametersError("Please configure the node")
        
        if len(self.output_column_name) > 100:
            raise knext.InvalidParametersError("Too Long Column Name")

        # Return output table schema
        return input_schema_1.append(knext.Column(knext.double(), name=self.output_column_name))

 
    def execute(self, exec_context, input_1, input_2):

        exec_context.set_warning("This node was developed on Friday evening.")

        # Convert to Pandas DataFrame
        trips_df = input_1.to_pandas() 
        city_locations_df = input_2.to_pandas()

        # Compute distances
        distances = self._compute_distances(trips_df, city_locations_df)
        trips_df[self.output_column_name] = distances

        # Generate view
        self._create_view(trips_df, city_locations_df, distances)
        
        # Return results
        return knext.Table.from_pandas(trips_df), knext.view_matplotlib()

    def _compute_distances(self, trips_df, city_locations_df):
        """ Return the distance of each trip in the trip_df, using the locations from the second input """
        coordinates_df = pd.DataFrame()
        coordinates_df[self.lat_column] = np.radians(city_locations_df[self.lat_column])
        coordinates_df[self.long_column] = np.radians(city_locations_df[self.long_column])

        array_with_city_names = city_locations_df[self.location_column].unique()

        dist = DistanceMetric.get_metric('haversine')
        dist_array = dist.pairwise(coordinates_df[[self.lat_column, self.long_column]].to_numpy())*6373
        distance_df = pd.DataFrame(dist_array, columns=array_with_city_names, index=array_with_city_names)

        from_city_col = trips_df.columns[0]
        to_city_col = trips_df.columns[1]

        distances = []
        for couple_city in zip(trips_df[from_city_col],trips_df[to_city_col]):
            dist = distance_df.loc[couple_city[0], couple_city[1]]
            LOGGER.info("The distance between "+ couple_city[0]+ " and "+ couple_city[1]+ " is "+ str(dist))
            distances.append(dist)
        
        return distances

    def _create_view(self, trips_df, city_locations_df, distances):
        """ Draw a diagram with all cities and the trips between them """
        cities = city_locations_df[self.location_column]
        from_city_col = trips_df.columns[0]
        to_city_col = trips_df.columns[1]
        x_line = trips_df[from_city_col]
        y_line = trips_df[to_city_col]

        _, ax = plt.subplots()
        plt.title(self.output_column_name)

        x_pixel = city_locations_df[self.long_column]
        y_pixel = city_locations_df[self.lat_column]

        stdx = np.std(x_pixel)/2
        stdy = np.std(y_pixel)/2

        maxX = max(x_pixel+stdx)
        maxY = max(y_pixel+stdy)
        minX = min(x_pixel-stdx)
        minY = min(y_pixel-stdy)

        for i, txt in enumerate(city_locations_df[self.location_column].tolist()):
            ax.annotate(txt, (x_pixel[i]-stdx/2, y_pixel[i]-stdy/2))

        plt.xlim([minX,maxX])
        plt.ylim([minY,maxY])

        for i in range(len(x_line)):
            LOGGER.info("Drawing: " + x_line[i] + " -> " + y_line[i])
            lats_longs_mini_df = city_locations_df.loc[(cities==x_line[i]) | (cities==y_line[i])]
            lats = lats_longs_mini_df[self.lat_column].tolist()
            longs = lats_longs_mini_df[self.long_column].tolist()
            plt.plot(longs, lats, 'ro-')

            mid_point_x = (longs[0] + longs[1])/2
            mid_point_y = (lats[0] + lats[1])/2

            ax.annotate(str(distances[i].round(2)) + " Km", (mid_point_x, mid_point_y))
