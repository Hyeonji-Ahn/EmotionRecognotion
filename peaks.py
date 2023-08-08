import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd

# Creating a sample from a bimodal distribution

csv_file_path = r"C:\Users\ahj28\Desktop\Garcia DISC Data\ControlsAfterAugust\Ketamin\\2a\graph\Whole Face.csv"  # Replace with the path to your CSV file
column_index_to_plot = 1  # Replace with the index of the column you want to plot (0 for the first column)

df = pd.read_csv(csv_file_path, header=None)

# Extract data from the specified column index
data = df.iloc[:, column_index_to_plot]

# Using KDE to estimate the distribution
kde = gaussian_kde(data)

# Generating a range of values to evaluate the KDE
x_eval = np.linspace(min(data), max(data), 1000)
density = kde(x_eval)

# Finding the peaks in the KDE, which correspond to the modes
peaks, _ = find_peaks(density)

# Print the location of the modes
for peak in peaks:
    print(f"Mode at x = {x_eval[peak]}")
    print(density[peak])

# Plotting the KDE and the peaks
plt.plot(x_eval, density)
plt.scatter(x_eval[peaks], density[peaks], color='red') # Peaks in red
plt.show()

