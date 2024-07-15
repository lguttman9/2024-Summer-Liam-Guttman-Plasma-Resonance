#Phase echoe plotter 
#Chat Mev-Reader
import struct 
import numpy as np
import math 
import matplotlib.pyplot as plt
from datetime import datetime as dt

with open(r"/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00/mev-2020-000-0007-38-00.dat", mode="rb") as file:
    file_content = file.read()
   
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
    phase_lists = [np.array([], dtype=int) for _ in range(c_tot)]
    for i in range(c_tot):
        idc[i] = struct.unpack("<h", file_content[152+i*52:154+i*52])[0]
        #i_off[i] = np.append(i_off[i],idc[i])
        qdc[i] = struct.unpack("<h", file_content[154+i*52:156+i*52])[0]
        #q_off[i] = np.append(q_off[i],qdc[i])
    #print(idc,qdc)
    phase_lists.append(np.array([],dtype = int))
#re-organize arrays so that theres an array for each column not each row
    for i in range(p_tot-1):
        for j in range(c_tot):
            i_lists[j] = np.append(i_lists[j], i_signal[i+1, j])
            q_lists[j] = np.append(q_lists[j], q_signal[i+1, j])
            #calculate and append amplitude 
            phase = np.arctan2(q_signal[i+1,j],i_signal[i+1,j]) - np.arctan2(q_signal[i,j],i_signal[i,j])
            phase_lists[j] = np.append(phase_lists[j], phase)
            if j == 6:
                phase_lists[7] = np.append(phase_lists[7],phase_lists[5][i]-phase_lists[6][i])
      
         


    #print(amp_lists)
    #print(i_lists[0], q_lists[0],amp_lists[0])
    file.close()
#print(time, type(time))
#a= dt.fromtimestamp(time)
#print(a,type(a))
#print(np.max(amp_lists[5]), np.where(amp_lists[5] == np.max(amp_lists[5])))
#print(np.max(i_lists[5]), np.max(q_lists[5]),np.where(i_lists[5] == np.max(i_lists[5])))
print(phase_lists[7])
fig, axes = plt.subplots(3)
axes[0].plot(phase_lists[5])
axes[1].plot(phase_lists[6])
axes[2].plot(phase_lists[7])
#plt.plot(amp_lists[5])
#plt.plot(amp_lists[6])
plt.show()

