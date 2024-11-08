import logging

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import Counter

def plt_3d(x, y, z, x_label, y_label, z_label):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a 3D surface
    ax.plot_surface(x[:, np.newaxis], y, z, cmap='viridis')  # Add [:, np.newaxis] to make x a column vector
    
    # Add labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # Display the plot
    plt.show()

def plt_2d(x, y, x_label, y_label):
    # Create a 2D plot
    #plt.figure(figsize=(6,6))
    plt.figure(figsize=(7,6))

    # Plot the relationship between x and y
    plt.plot(x, y, marker='o', linestyle='-')  # Different markers and linestyles can be selected

    # Add labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add labels and set font size
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    # Set the font size of the ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Adjust the margins to ensure the y-axis label is fully visible
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    # Display the plot
    plt.grid(True)  # Add a grid
    plt.show()

def plt_3d_comparison(x, y, z1, z2, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid
    X, Y = np.meshgrid(x, y)

    # Plot the z1 surface
    ax.plot_surface(X, Y, z1.T, cmap='viridis', alpha=0.7, label="Z1")  # Transpose z1 to match the shape of X and Y
    # Plot the z2 surface
    ax.plot_surface(X, Y, z2.T, cmap='plasma', alpha=0.7, label="Z2")  # Transpose z2 to match the shape of X and Y

    # Add labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Adjust the view angle
    ax.view_init(elev=10, azim=30)

    # Display the plot
    plt.show()

def plt_multi_line(x, datas, legends, x_label, y_label, y_bottoms, legend_label, legend_loc, fig_size):
    font_size = 20
    plt.figure(figsize=fig_size)

    logging.info(f"datas:{datas}")
    logging.info(f"legends:{legends}")
    # Iterate through each inner list in result and embedding_rate to plot the curves
    for i, (y_values, legend) in enumerate(zip(datas, legends)):
        if (len(legend_label) > 0):
            plt.plot(x, y_values, label=f'{legend_label}={legend}')
        else:
            plt.plot(x, y_values, label=f'{legend}')

    # Add a title and labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title, fontsize=20)
    plt.legend(loc=legend_loc, fontsize=font_size)

    # Add labels and set font size
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    # Set the font size of the ticks
    #plt.xticks(range(int(min(x)), int(max(x))+1), fontsize=font_size)  # Display only integer values on the x-axis ticks
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    if (len(y_bottoms)>0):
        plt.ylim(bottom=y_bottoms[0])

    # Adjust the margins to ensure the y-axis label is fully visible
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
    #plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    # Show the plot
    plt.show()

def draw_graph(G):
    # Plot graphs
    plt.figure(figsize=(8, 6))  # Set the size of the plot
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # Add title
    plt.title("Graph G")

    # Display the plot
    plt.show()
def draw_embedding(G, query, mapping):
    plt.figure(figsize=(10, 6))

    # 绘制大图 G
    pos_G = nx.spring_layout(G)  # Layout
    nx.draw(G, pos_G, with_labels=True, node_color='lightblue', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # Highlight the mapped nodes
    for g_node, h_node in mapping.items():
        plt.scatter(*pos_G[g_node], color='red', s=200)  # Mark the nodes in the large graph with red
        plt.text(pos_G[g_node][0], pos_G[g_node][1], f" {h_node}", color='white', fontsize=12, ha='center',
                 va='center')  # Display the subgraph node labels

    # Plot the query graph
    pos_H = nx.spring_layout(query)
    nx.draw(query, pos_H, with_labels=True, node_color='lightgreen', node_size=700, font_size=16, font_color='black',
            font_weight='bold', edge_color='gray')

    # Add a title
    plt.title("Graph G with Mapping to Subgraph query")
    plt.show()
def plt_his(sorted_data):
    print(f"sorted_data:{sorted_data}")
    indices = np.arange(len(sorted_data))
    # Plot a bar chart
    plt.figure(figsize=(10, 6))
    #plt.hist(indices, sorted_data, color='lightblue', edgecolor='black', alpha=0.7)  # Set the color and edges of the bar chart
    plt.plot(indices, sorted_data, marker='o', color='lightblue', linestyle='-', linewidth=2, markersize=5)
    plt.title('Histogram of Normally Distributed Data (N(100, 50))')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)  # Add a grid
    plt.show()
def plt_distribution(node_types):

    # Calculate the distribution of node_type
    node_type_count = Counter(node_types)

    # Print the distribution of node_type
    print("Node type distribution:", node_type_count)

    # Visualize the distribution
    plt.bar(node_type_count.keys(), node_type_count.values(), color='lightblue')
    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.title('Node Type Distribution in Graph G')
    plt.show()

def plt_viridis_by_log(FSI_elapses, ORI_elapses, legends, x_label, y_label, title):
    # Take the logarithm (base 10) of the data
    FSI_log = [np.log10(row) for row in FSI_elapses]
    ORI_log = [np.log10(row) for row in ORI_elapses]

    # Create a scatter plot
    plt.figure(figsize=(6, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(FSI_log)))

    for i, (fsi_row, ori_row, legend) in enumerate(zip(FSI_log, ORI_log, legends)):
        plt.scatter(fsi_row, ori_row, color=colors[i], label=legend)

    # Set axis labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Set the x-axis and y-axis range from -3 to 3
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Display the legend
    plt.legend(loc="best")

    # Display the plot
    plt.show()
def plt_viridis(FSI, ORI, legends, x_label, y_label, title):
    # Create a plot
    plt.figure(figsize=(8, 8))

    # Generate different colors using the viridis colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(FSI)))

    # Iterate over each data group and plot a scatter plot
    for i, (fsi_row, ori_row, legend) in enumerate(zip(FSI, ORI, legends)):
        plt.scatter(fsi_row, ori_row, color=colors[i], label=legend)

    # Set the x-axis and y-axis ranges to be consistent, such as from -3 to 3 (modify if needed)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('equal')  # Keep the aspect ratio of the x and y axes consistent

    # Set axis labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Display the legend
    plt.legend(loc="best")

    # Display the plot
    plt.show()
def plt_double_bars(datas, legends, x_label, y_label, title):
    plt.figure(figsize=(6,5))
    # Extract data
    keys = list(datas.keys())  # x-axis
    values = list(datas.values())  # Each value is a tuple

    # Split the two elements of each tuple into separate lists
    val1 = [v[0] for v in values]  # First element
    val2 = [v[1] for v in values]  # Second element

    # Set the width and position of the bars
    x = np.arange(len(keys))  # Position of each key
    bar_width = 0.35  # Width of each bar
    font_size = 20

    # Plot a bar chart
    plt.bar(x - bar_width/2, val1, width=bar_width, label=legends[0])
    plt.bar(x + bar_width/2, val2, width=bar_width, label=legends[1])

    # Add labels and a title
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    #plt.title(title, fontsize=font_size)
    plt.xticks(x, keys, fontsize=font_size)  # Set the x-axis ticks to the values of keys
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

    # Display the plot
    plt.show()