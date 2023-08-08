import pandas as pd
import matplotlib.pyplot as plt

def plot_density_graph(csv_file, column_index):
    # Read the CSV file using pandas with no header
    df = pd.read_csv(csv_file, header=None)

    # Extract data from the specified column index
    data = df.iloc[:, column_index]

    # Create a density plot using matplotlib
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.title(f"Density Graph of Column {column_index}")
    plt.xlabel(f"Column {column_index}")
    plt.ylabel("Density")
    plt.hist(data, density=True, bins=30, alpha=0.7, color='g')  # You can adjust the number of bins and color
    plt.grid(True)
    plt.show()

# Example usage:
csv_file_path = r"C:\Users\ahj28\Desktop\Garcia DISC Data\ControlsAfterAugust\Ketamin\\2a\graph\Whole Face.csv"  # Replace with the path to your CSV file
column_index_to_plot = 1  # Replace with the index of the column you want to plot (0 for the first column)
plot_density_graph(csv_file_path, column_index_to_plot)