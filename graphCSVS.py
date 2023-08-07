import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_csv_files(files_list, x_column_index, y_column_index, labels):
    # Initialize the figure and axis
    fig, ax = plt.subplots()

    # Define a list of colors for each line
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

    # Loop through each CSV file and plot the data
    for i, (file, label) in enumerate(zip(files_list, labels)):
        file_path = os.path.join(r'C:\Users\ahj28\Desktop\Garcia DISC Data\ControlsAfterAugust', file)
        df = pd.read_csv(file_path, header=None)  # Skip header row with column names
        color = colors[i % len(colors)]  # Cycle through colors if more lines than colors
        ax.plot(df.iloc[:, x_column_index], df.iloc[:, y_column_index], label=label, color=color)

    # Set labels and legend
    ax.set_xlabel("Frames")
    ax.set_ylabel("Values")
    ax.legend()

    plt.vlines(x=np.linspace(0,110,10), ymin=0, ymax=len(np.linspace(1,21,20)), colors='purple', ls='--', lw=1)

    # Show the plot
    plt.show()

    

# Example usage with three files
files_list = ['b1\\b1bef\graph\Whole Face.csv', 'b1\\b1aft\graph\Whole Face.csv', 'Ketamin\\1b\graph\Whole Face.csv']  # Add more filenames as needed
x_column_index = 0  # Assuming the x values are in the first column (index 0)
y_column_index = 1  # Assuming the y values are in the second, third, and fourth columns (indices 1, 2, and 3)
labels = ['Control before','Control after','Pre-esketamine']
0
plot_csv_files(files_list, x_column_index, y_column_index, labels)
