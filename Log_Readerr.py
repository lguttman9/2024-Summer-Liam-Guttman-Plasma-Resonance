#Mev.log reader (again?)
#@liamguttman
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Open and read the file
def file_reader(file_path):
    openfile = open(file_path)
    file = openfile.readlines()

        # Skip the first 17 lines
    c = 0
    d = 0
    while c >= d or d<=15:
        if file[c][0] == '#':
            c+=1
        d+=1
         
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

snr = file_reader("/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00.log")
array_list = file_reader("/srv/public/pbrown/fullwave/multi-2020-000-29.log")

directory = '/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00'
plots = [np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int),np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)]

# Get the list of .dat files and sort them in numeric order
files = [f for f in os.listdir(directory) if f.endswith('.dat')]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

file_count = 0
max_files = 4000 # Limit to first 2000 files
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
        hex_path = int(file_path[63:67], 16)
        if file_path[63:67] in letter: 
            for i in range(len(array_list[2])):
                    
                if hex_path == array_list[2][i]:
                    for j in range(len(snr[0])):
                        if array_list[2][i] == snr[0][j]:
                            if float(snr[5][j]) > 10:
                                plots[0] = np.append(plots[0],array_list[2][i])
                                plots[1] = np.append(plots[1], snr[0][j])
                                plots[2] = np.append(plots[2], snr[5][j])

#print(snr[5][:15],snr[0][:15],array_list[2][:15])
print(plots[0][:10],plots[1][:10],plots[2][:10])

#"/srv/meteor/radar/spool/mev-38/mev-2020-000-38-00.log"
#"/srv/public/pbrown/fullwave/multi-2020-000-29.log"



