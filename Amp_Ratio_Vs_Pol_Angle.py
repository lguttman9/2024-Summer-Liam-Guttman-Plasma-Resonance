# Amp_Ratio_Vs_Pol_Angle
from ast import literal_eval as unhex
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def file_reader(file_path):
    openfile = open(file_path)
    file = openfile.readlines()
        # Skip the first 17 lines
    c = 0
    d = 0
    while c >= d or d<=15:
        if file[c][0] == '#':
            c+=1
            #print(c)
        d+=1
        #print(d)
    #print(c,d) 
    #print(file[c])
    openfile.close()
    lines = file[c:]
   
    array_list = []

    # Outer loop to iterate over all lines in the file
    for a in range(len(lines)):
        b = 0  # Reset b for each line
        data = lines[a].split(' ') #splits the strings in a line with a space
        i=0
        for element in data:
            if element != '' and element != '#' and element != '':
                # Ensure the array_list has enough arrays
                
                while len(array_list) <= b:
                    #append empty list for each column
                    array_list.append(np.array([], dtype=float))
                if element == '-----':
                    #append zero if theres no tag on a certain frequency
                    array_list[b] = np.append(array_list[b], 0.0)
                    i+=1
                else: 
                    try:
                        # Convert to integer and append to the numpy array
                        if i<=2:
                            #If its one fo the first three columns append a base 16 int (hexidecimal)
                            array_list[b] = np.append(array_list[b],int(element,16))
                            i+=1
                        else:
                            #otherwise append regular int 
                            array_list[b] = np.append(array_list[b], float(element))
                    except ValueError:
                        # Handle the case where the conversion to int fails
                    
                        continue
                    
                b += 1
    return array_list
array_list = file_reader('/srv/public/pbrown/fullwave/multi-2020-000-29.log')
snr = file_reader("/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00.log")
# Gain Calculator
def GtGr(tht, phi, th):
    phi = phi + th + 90
    if (phi > 360):
        phi = phi - 360
    GTbroad = 5.5781236 - 2.8200065 * np.cos(np.radians(tht))**2 - 3.5209044 * np.cos(np.radians(tht))**4 + 10.400722 * np.cos(np.radians(tht))**6  - 9.5485295 *np.cos(np.radians(tht))**8
    GTbroaddB = 10*np.log10(GTbroad)
    GRbroad = 4.1036476 + 1.5400743 * np.cos(np.radians(tht))**2 -3.1886833 * np.cos(np.radians(tht))**4 + 2.9979406 * np.cos(np.radians(tht))**6 - 5.3852557 *np.cos(np.radians(tht))**8
    GRbroaddB = 10*np.log10(GRbroad)
    if (phi < 90 or phi > 270):
        GTenddB = (0.031795133 + 4.8990288 * np.cos(np.radians(tht))**2 + 13.669295 * np.cos(np.radians(tht))**4 - 25.170221 * np.cos(np.radians(tht))**6 + 22.814163 *np.cos(np.radians(tht))**8)/(7.7827215 - 8.1684971 + 11.627713)
        if (GTenddB < 0):
            GTenddB = 0
        GRenddB = (-0.006819758 + 7.9024553 *np.cos(np.radians(tht))**2 -1.538963 * np.cos(np.radians(tht))**4 + 12.2719 * np.cos(np.radians(tht))**6 -4.0899217 *np.cos(np.radians(tht))**8)/(8.2011129 - 9.6857106 + 13.001805)
        if (GRenddB < 0):
            GRenddB = 0
        GTenddB = np.cos(np.radians(phi))**2 * GTenddB**0.6
        GRenddB = np.cos(np.radians(phi))**2 * GRenddB**0.65
        GTtdB = GTbroaddB - (7.7827215 * GTenddB - 8.1684971 * GTenddB**2 + 11.627713 * GTenddB**3)
        GRtdB = GRbroaddB - (8.2011129 * GTenddB - 9.6857106 * GTenddB**2 + 13.001805 * GTenddB**3)
    else:
        GTenddB = (-0.021062092 + 7.1497901 *np.cos(np.radians(tht))**2 - 3.3448948 * np.cos(np.radians(tht))**4 + 13.533221 * np.cos(np.radians(tht))**6 - 4.7793303 *np.cos(np.radians(tht))**8)/(6.4776762 - 4.5794876 + 8.1254215)
        if (GTenddB < 0):
            GTenddB = 0
        GRenddB = (-0.005270753 + 6.3391005 *np.cos(np.radians(tht))**2 + 3.3638038 * np.cos(np.radians(tht))**4 + 0.36978112 * np.cos(np.radians(tht))**6 + 2.3794633 *np.cos(np.radians(tht))**8)/(6.2636256 - 3.9807794 + 7.4747311)
        if (GRenddB < 0):
            GRenddB = 0
        GTenddB = np.cos(np.radians(phi))**2 * GTenddB**0.6
        GRenddB = np.cos(np.radians(phi))**2 * GRenddB**0.65
        GTtdB = GTbroaddB - (6.4776762 * GTenddB - 4.5794876 * GTenddB**2 + 8.1254215 * GTenddB**3)
        GRtdB = GRbroaddB - (6.2636256 * GTenddB - 3.9807794 * GTenddB**2 + 7.4747311 * GTenddB**3)
    GTt = 10**(GTtdB/10)
    GRt = 10**(GRtdB/10)
    Gt = GTt*GRt
    return (GRt)

# Chat Mev-Reader
import struct
import math
import os
import numpy as np

directory = '/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00'
plots = [np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int)]

# Get the list of .dat files and sort them in numeric order
files = [f for f in os.listdir(directory) if f.endswith('.dat')]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

file_count = 0
max_files = 2000 # Limit to first 2000 files
letter = ['0001', '0006', '0007', '0009', '000F', '0013', '0020','0021','0022'
          ,'0027', '002E', '0033', '0034', '003A', '003D', '003E', '0041', '004A'
          ,'004B','004D', '004F', '0053', '0058', '005B', '0064', '0067', '0069', '0075'
          , '0078', '0079', '007C', '0086', '0089', '0092', '0093', '0094', '0098'
          , '0099', '00A9', '00AA', '00AC', '00B2','00B7', '00BC', '00BE', '00C0'
          , '00C4', '00C5', '00C9', '00CB', '00D1','00DA','00DD','00E0','00E2','00EA'
          ,'00F2','0102','10F','0114','0116','0118','0119','011C','011E','0126',
          '0127','012A','012B','0139','013B','013C','0140','0145','0147','0150',
          '0155','0159','015D','0165','0168','0177','0188','018A','019B'
          ]

for filename in files:
    if filename.endswith('.dat'):
        file_path = os.path.join(directory, filename)
        #if file_path[63:67] in letter:    
        with open(file_path, 'rb') as f:  # Open the file in binary mode
            file_content = f.read()

            # Calculate the total number of bytes
            num_bytes = len(file_content)

            # Get the number of bytes for each part of the MEV
            header_size = 76  # Fixed for all MEV
            info_size = struct.unpack("I", file_content[44:48])[0]
            chan_size = struct.unpack("I", file_content[48:52])[0]
            time = struct.unpack("q", file_content[84:92])[0]

            # Give you the total number of channels and pulses
            c_tot = struct.unpack("i", file_content[52:56])[0]
            p_tot = struct.unpack("i", file_content[56:60])[0]

            # Give you the file tag at the start (22431053 is new mev file)
            mev = struct.unpack("i", file_content[0:4])[0]

            # Calculate the number of bytes in the header
            meta_size = header_size + info_size + chan_size * c_tot

            # Calculate the size of 'i' and 'q' values in bytes
            data_size = num_bytes - meta_size

            i_signal = np.zeros((p_tot, c_tot), dtype=int)
            q_signal = np.zeros((p_tot, c_tot), dtype=int)
            idc = np.zeros((c_tot), dtype=int)
            qdc = np.zeros((c_tot), dtype=int)

            for c in range(c_tot):
                for p in range(p_tot):
                    i_signal[p, c] = struct.unpack("<h", file_content[meta_size + 2 * (c * p_tot + p):meta_size + 2 * (c * p_tot + p) + 2])[0]
                    q_signal[p, c] = struct.unpack("<h", file_content[meta_size + 2 * ((c + c_tot) * p_tot + p):meta_size + 2 * ((c + c_tot) * p_tot + p) + 2])[0]

            i_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            q_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            amp_lists = [np.array([], dtype=int) for _ in range(c_tot)]
            for i in range(c_tot):
                idc[i] = struct.unpack("<h", file_content[152 + i * 52:154 + i * 52])[0]
                qdc[i] = struct.unpack("<h", file_content[154 + i * 52:156 + i * 52])[0]

            for i in range(p_tot):
                for j in range(c_tot):
                    i_lists[j] = np.append(i_lists[j], i_signal[i, j])
                    q_lists[j] = np.append(q_lists[j], q_signal[i, j])
                    amplitude = math.sqrt(i_signal[i, j] ** 2 + q_signal[i, j] ** 2)
                    amp_lists[j] = np.append(amp_lists[j], amplitude)

            f.close()
            
            hex_path = int(file_path[63:67], 16)

            top1 = np.where(amp_lists[5] == np.max(amp_lists[5]))[0][0]
            max_avg = np.array(amp_lists[5][top1-5:top1+5]).mean() / np.array(amp_lists[6][top1-5:top1+5]).mean()

            dec_time = 0
            time_list = []
            
            for i in range(len(amp_lists)):
                avg = np.mean(amp_lists[i])
                for j in range(len(amp_lists[i])):
                    if amp_lists[i][j] > avg:
                        dec_time += 1
                    else:
                        time_list.append(dec_time)
                        dec_time = 0

                
                #for i in range(len(time_list)):
                #if time_list[i]:
                hex_path = int(file_path[63:67], 16)

 
                for i in range(len(array_list[2])):
                        #print(file_path[63:67],array_list[2][i], array_list[4][i],snr[5][i-1])
                        if hex_path == array_list[2][i]:    
                            for j in range(len(snr[0])):    
                                if array_list[2][i] == snr[0][j]:
                                    if snr[5][j] >= 15: #and snr[5][j]<=20:
                                        if np.max(time_list) >= 90:

                                        
                                            plots[11] = np.append(plots[11],np.max(time_list))
                                            plots[10] = np.append(plots[10],snr[5][j])
                                            gain_ratio = GtGr(array_list[11][i], array_list[12][i], 254) / GtGr(array_list[11][i], array_list[12][i], 344)
                                            plots[9] = np.append(plots[9], time)
                                            plots[8] = np.append(plots[8], array_list[6][i])
                                            plots[7] = np.append(plots[7],array_list[7][i])
                                            plots[6] = np.append(plots[6],array_list[10][i])
                                            plots[5] = np.append(plots[5], array_list[8][i])
                                            plots[4] = np.append(plots[4], array_list[12][i])
                                            plots[3] = np.append(plots[3], array_list[11][i])
                                            plots[2] = np.append(plots[2], np.log(max_avg * gain_ratio))
                                            plots[1] = np.append(plots[1], hex_path)
                                            plots[0] = np.append(plots[0], array_list[4][i])
                                
                            

        file_count += 1
        if file_count >= max_files:
            break

def remove_outliers_zscore(x, y, threshold=3):
    # Calculate Z-scores
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    
    z_scores_x = (x - mean_x) / std_x
    z_scores_y = (y - mean_y) / std_y
    
    # Filter out points with a Z-score greater than the threshold
    mask = (np.abs(z_scores_x) < threshold) & (np.abs(z_scores_y) < threshold)
    return x[mask], y[mask]


# Function to remove outliers using the IQR method
def remove_outliers_iqr(x, y, multiplier=1.5):
    # Calculate IQR for x
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    lower_bound_x, upper_bound_x = q1_x - multiplier * iqr_x, q3_x + multiplier * iqr_x
    
    # Calculate IQR for y
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    lower_bound_y, upper_bound_y = q1_y - multiplier * iqr_y, q3_y + multiplier * iqr_y
    
    # Filter out points outside of IQR range
    mask = (x > lower_bound_x) & (x < upper_bound_x) & (y > lower_bound_y) & (y < upper_bound_y)
    return x[mask], y[mask]

df = pd.DataFrame(plots).T
df1 = df[(df[0]>35)]
y1 = df1[2]
x1 = df1[0]

y = df[2]
x = df[0]

print(array_list[2][:10],array_list[4][:10])
print(plots[2][:10])
print(plots[0][:10])
print(plots[1][:10])
print(plots[11][:10])

print(df,df1,df1[0],df1[2])



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
    plt.show()
except np.linalg.LinAlgError as e:
    print(f"LinAlgError during fitting: {e}")
#c=c_clean, cmap='Purples',
