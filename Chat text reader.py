import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open and read the file
with open("/home/meteor/meteor/pulses-20200320-102508-0001-38-00.txt") as openfile:
    file = openfile.readlines()

# Skip the first 17 lines
lines = file[17:]

# Initialize an empty list to store arrays
array_list = []

# Outer loop to iterate over all lines in the file
for a in range(len(lines)):
    b = 0  # Reset b for each line
    data = lines[a].split()

    for element in data:
        if element != '' and element != '#' and element != 'n':
            # Ensure the array_list has enough arrays
            while len(array_list) <= b:
                array_list.append(np.array([], dtype=float))
            try:
                # Convert to integer and append to the numpy array
                array_list[b] = np.append(array_list[b], float(element))
            except ValueError:
                # Handle the case where the conversion to int fails
                continue
            b += 1

# Remove empty arrays
array_list = [arr for arr in array_list if arr.size > 0]

#print("Final arrays:", array_list)
#print(f"Total arrays: {len(array_list)}")
print(array_list[0][0], len(array_list))
fig, axes = plt.subplots(2)
#create subplot table 2rows by 3 columns
axes[0].plot(array_list[3])
#axes[1].plot(array_list[7])
#axes[2].plot(array_list[11])
#axes[3].plot(array_list[15])
#axes[4].plot(array_list[19])
#axes[5].plot(array_list[23])
#axes[6].plot(array_list[27])
plt.show()

