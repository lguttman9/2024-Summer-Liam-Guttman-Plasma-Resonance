#Finale
from ast import literal_eval as unhex
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval as unhex

int_value = int('000B',16)
print(int_value)



openfile = open("/srv/public/pbrown/fullwave/multi-2020-000-29.log")
file = openfile.readlines()


array_list = []

# Outer loop to iterate over all lines in the file
for a in range(len(file)):
    b = 0  # Reset b for each line
    data = file[a].split(' ')
    i=0
    for element in data:
        if element != '' and element != '#' and element != '':
            # Ensure the array_list has enough arrays
            
            while len(array_list) <= b:
                array_list.append(np.array([], dtype=float))
            if element == '-----':
                array_list[b] = np.append(array_list[b], 0.0)
                i+=1
            else: 
                try:
                    # Convert to integer and append to the numpy array
                    if i<=2:
                        array_list[b] = np.append(array_list[b],int(element,16))
                        i+=1
                    else:
                        array_list[b] = np.append(array_list[b], float(element))
                except ValueError:
                    # Handle the case where the conversion to int fails
                   
                    continue
                
            b += 1

#print(array_list[0][:10],len(file))


#Chat Mev-Reader
import struct 
import math 
import os
import numpy as np

directory = '/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00'
max_amplitudes = np.array([], dtype=float)  # Rename the variable to avoid conflict

# Get the list of .dat files and sort them in numeric order
files = [f for f in os.listdir(directory) if f.endswith('.dat')]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

file_count = 0
max_files = 5  # Limit to first 5 files

for filename in files:
    if filename.endswith('.dat'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:  # Open the file in binary mode
            array_list = []
            file_content = f.read()

            # Calculate the total number of bytes
            num_bytes = len(file_content)

            # Get the number of bytes for each part of the MEV
            header_size = 76  # Fixed for all MEV
            info_size = struct.unpack("I", file_content[44:48])[0]
            chan_size = struct.unpack("I", file_content[48:52])[0]

            # Give you the total number of channels and pulses
            c_tot = struct.unpack("i", file_content[52:56])[0]
            p_tot = struct.unpack("i", file_content[56:60])[0]

            # Give you the file tag at the start (22431053 is new mev file)
            mev = struct.unpack("i", file_content[0:4])[0]

            # Print the unpacked header data
            #print("Header Information:")
            #print(f"c_tot: {c_tot}, p_tot: {p_tot}, MEV: {mev}")

            # Calculate the number of bytes in the header
            meta_size = header_size + info_size + chan_size * c_tot

            # Calculate the size of 'i' and 'q' values in bytes
            data_size = num_bytes - meta_size

            # Print the i and q signals
            i_signal = np.zeros((p_tot, c_tot), dtype=int)
            q_signal = np.zeros((p_tot, c_tot), dtype=int)
            idc = np.zeros((c_tot), dtype=int)
            qdc = np.zeros((c_tot), dtype=int)

            for c in range(c_tot):
                for p in range(p_tot):
                    i_signal[p, c] = struct.unpack("<h", file_content[meta_size + 2 * (c * p_tot + p):meta_size + 2 * (c * p_tot + p) + 2])[0]
                    q_signal[p, c] = struct.unpack("<h", file_content[meta_size + 2 * ((c + c_tot) * p_tot + p):meta_size + 2 * ((c + c_tot) * p_tot + p) + 2])[0]

            # Convert to lists
            i_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            q_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            amp_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            for i in range(c_tot):
                idc[i] = struct.unpack("<h", file_content[152 + i * 52:154 + i * 52])[0]
                qdc[i] = struct.unpack("<h", file_content[154 + i * 52:156 + i * 52])[0]

            #print(idc, qdc)

            for i in range(p_tot):
                for j in range(c_tot):
                    i_lists[j] = np.append(i_lists[j], i_signal[i, j])
                    q_lists[j] = np.append(q_lists[j], q_signal[i, j])
                    amplitude = math.sqrt(i_signal[i, j] ** 2 + q_signal[i, j] ** 2)
                    amp_lists[j] = np.append(amp_lists[j], amplitude)

            f.close()
            # Append the maximum amplitude from channel 6 to max_amplitudes array
            max_amplitudes = np.append(max_amplitudes, np.max(amp_lists[6]))

        file_count += 1
        if file_count >= max_files:
            break  # Stop after processing the first 5 files
    print(file_path)

print(max_amplitudes)

            


'''

print(max(amp_lists[6]))
#fig, axes = plt.subplots(2)
#axes[0].plot(amp_lists[0])
plt.plot(amp_lists[0])
'''

