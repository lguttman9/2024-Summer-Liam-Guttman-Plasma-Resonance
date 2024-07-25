#data_frame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[5,6,7],[1,2,3],[7,8,9]],dtype = int)

print(data[0])
df = pd.DataFrame(data).T
print(df)

df1 = df[(df[0]>5)]
print(df1,df1[0],df1[1])

y = df1[1]
x = df1[0]


# Check for NaNs or infinite values in x and y
if np.isnan(x).any() or np.isnan(y).any():
    print("NaNs detected in data.")
if np.isinf(x).any() or np.isinf(y).any():
    print("Infinite values detected in data.")

# Remove NaNs or infinite values

valid_mask = np.isfinite(x) & np.isfinite(y)
x_clean = x[valid_mask]
y_clean = y[valid_mask]

# Ensure all arrays have the same length
min_length = min(len(x_clean), len(y_clean))

x_clean = x_clean[:min_length]
y_clean = y_clean[:min_length]


# Scale the data
x_mean = x_clean.mean()
x_std = x_clean.std()
y_mean = y_clean.mean()
y_std = y_clean.std()

x_scaled = (x_clean - x_mean) / x_std
y_scaled = (y_clean - y_mean) / y_std

# Fit the scaled data
try:
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # Plot the results
    plt.scatter(x, y,  alpha=0.5)
    plt.plot(x, p(x) , "r", label='Fitted line')  # Adjust the line for the original scale
    plt.axhline(y=0, color='g', label="Zero line")
    plt.xlabel("Polarization Angle")
    plt.ylabel("log of Amplitude ratio (A6/A7)")
    plt.title("log of Max Amplitude Ratio vs Polarization Angle of 300-400 overdense Meteors with length >150 points, with SNR greater then 20,  (accounting for gain patterns)")
    plt.legend()
    plt.show()
except np.linalg.LinAlgError as e:
    print(f"LinAlgError during fitting: {e}")
#c=c_clean, cmap='Purples',

