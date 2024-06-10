#.log file Reader
from ast import literal_eval as unhex
import numpy as np
import matplotlib.pyplot as plt 
from ast import literal_eval as unhex

int_value = int('000B',16)
print(int_value)


#opens and reads the file
openfile = open("/srv/public/pbrown/fullwave/multi-2020-000-29.log")
file = openfile.readlines()


array_list = []

# Outer loop to iterate over all lines in the file
for a in range(len(file)):
    b = 0  # Reset b for each line
    data = file[a].split(' ') #splits the strings in a line with a space
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

print(array_list[0][:10],len(file))

