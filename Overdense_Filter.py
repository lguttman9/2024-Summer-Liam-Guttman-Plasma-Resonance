#Overdense_Filter
#Chat Mev-Reader
import struct 
import numpy as np
import matplotlib.pyplot as plt
import os

directory = '/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00'
plots = [np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)]

# Get the list of .dat files and sort them in numeric order
files = [f for f in os.listdir(directory) if f.endswith('.dat')]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

file_count = 0
max_files = 10 # Limit to first 2000 files
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

max_times = []

for filename in files:
    if filename.endswith('.dat'):
        file_path = os.path.join(directory, filename)
        if file_path[63:67] in letter:    
            with open(file_path, 'rb') as f:  # Open the file in binary mode
                file_content = f.read()

                # Calculate the total number of bytes
                num_bytes = len(file_content)

                # Get the number of bytes for each part of the MEV
                header_size = 76 # Fixed for all MEV
                time = struct.unpack("q", file_content[84:92])[0]
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
                data_size = (num_bytes - meta_size)

                # Print the i and q signals
                i_signal = np.zeros((p_tot, c_tot), dtype=int)
                q_signal = np.zeros((p_tot, c_tot), dtype=int)
                idc = np.zeros((c_tot), dtype= int)
                qdc = np.zeros((c_tot), dtype= int)
            #get c_tot and p_tot from mev file 
                for c in range(c_tot):
                    for p in range(p_tot):          
                        i_signal[p, c] = struct.unpack("<h", file_content[meta_size+2*(c*p_tot+p):meta_size+2*(c*p_tot+p)+2])[0]
                        q_signal[p, c] = struct.unpack("<h", file_content[meta_size+2*((c+c_tot)*p_tot+p):meta_size+2*((c+c_tot)*p_tot+p)+2])[0]

                # Convert to lists
                i_lists = [np.array([], dtype=int) for _ in range(c_tot)]
                q_lists = [np.array([], dtype=int) for _ in range(c_tot)]
                i_off = [np.array([], dtype = int) for _ in range(c_tot)]
                q_off = [np.array([], dtype = int) for _ in range(c_tot)]
                amp_lists = [np.array([], dtype=int) for _ in range(c_tot)]
                for i in range(c_tot):
                    idc[i] = struct.unpack("<h", file_content[152+i*52:154+i*52])[0]
                    #i_off[i] = np.append(i_off[i],idc[i])
                    qdc[i] = struct.unpack("<h", file_content[154+i*52:156+i*52])[0]
                    #q_off[i] = np.append(q_off[i],qdc[i])
                #print(idc,qdc)
            #re-organize arrays so that theres an array for each column not each row
                for i in range(p_tot):
                    for j in range(c_tot):
                        i_lists[j] = np.append(i_lists[j], i_signal[i, j])
                        q_lists[j] = np.append(q_lists[j], q_signal[i, j])
                        #calculate and append amplitude 
                        amplitude = np.sqrt(i_signal[i,j]**2 + q_signal[i,j]**2)
                        amp_lists[j] = np.append(amp_lists[j], amplitude)
                        

                
                #print(amp_lists)
                #print(i_lists[0], q_lists[0],amp_lists[0])
                f.close()
            dec_time = 0
            time_list = [[] for _ in range(c_tot)]
            
            for i in range(len(amp_lists)):
                avg = np.mean(amp_lists[i])
                for j in range(len(amp_lists[i])):
                    if amp_lists[i][j] > avg:
                        dec_time += 1
                    else:
                        time_list[i].append(dec_time)
                        dec_time = 0

            while len(max_times) < len(time_list):
                max_times.append(np.array([], dtype=int))
            
            for i in range(len(time_list)):
                if time_list[i]:
                    max_times[i] = np.append(max_times[i], np.max(time_list[i]))
#print(np.max(amp_lists[5]), np.where(amp_lists[5] == np.max(amp_lists[5])))
#print(np.max(i_lists[5]), np.max(q_lists[5]),np.where(i_lists[5] == np.max(i_lists[5])))
print(max_times)
for i in range(len(max_times)):
    print(np.min(max_times[i]),np.mean(max_times[i]),np.median(max_times[i]))
fig, axes = plt.subplots(2)
axes[0].plot(amp_lists[5])
axes[1].plot(amp_lists[6])
#plt.plot(amp_lists[5])
#plt.plot(amp_lists[6])
plt.show()
