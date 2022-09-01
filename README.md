# **Create your KNIME Nodes using Python** 


Welcome to the KNIME Extension-Example! 

With this repository, we showcase how a KNIME node can be developed using Python. 

In this example we build the KNIME node called "**Geo Distances**" to compute the distance between two locations using their GPS coordinates.   

This example is adopted in the article "[4 Steps for your Python Team to Develop KNIME Nodes](https://www.knime.com/blog/4-steps-for-your-python-team-to-develop-knime-nodes)" on KNIME Blog.

More detailed documentation at "[Create a New Python based KNIME Extension](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#introduction)" on KNIME Docs.


![We implemented nodes to compute distances. The top input lists in two columns the locations between which a distance needs to be computed. The second input lists the coordinates for each location mentioned in the top input. The output is the first input with the distances in kilometers attached in a new third column.](README-figure.png "The KNIME node defined in this Python repository")
