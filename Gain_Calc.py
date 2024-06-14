
#.LOg REader

from ast import literal_eval as unhex
import numpy as np
import matplotlib.pyplot as plt 


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

#print(array_list[11][:10],array_list[12][:10],len(file))

#Gain Calc

def GtGr(tht,phi,th):
     phi = phi + th + 90
     if (phi>360):
         phi = phi - 360
     GTbroad = 5.5781236 - 2.8200065 * np.cos(np.radians(tht))**2 - 3.5209044 * np.cos(np.radians(tht))**4+ 10.400722 * np.cos(np.radians(tht))**6  - 9.5485295 *np.cos(np.radians(tht))**8
     GTbroaddB = 10*np.log10(GTbroad)
     GRbroad = 4.1036476 + 1.5400743 * np.cos(np.radians(tht))**2 -3.1886833 * np.cos(np.radians(tht))**4+ 2.9979406 * np.cos(np.radians(tht))**6 - 5.3852557 *np.cos(np.radians(tht))**8
     GRbroaddB = 10*np.log10(GRbroad)
     if (phi < 90 or phi > 270):
         GTenddB = (0.031795133 + 4.8990288 * np.cos(np.radians(tht))**2+ 13.669295 * np.cos(np.radians(tht))**4- 25.170221 * np.cos(np.radians(tht))**6 + 22.814163 *np.cos(np.radians(tht))**8)/(7.7827215 - 8.1684971 + 11.627713)
         if (GTenddB<0):
             GTenddB=0
         GRenddB = (-0.006819758 + 7.9024553 *np.cos(np.radians(tht))**2 -1.538963 * np.cos(np.radians(tht))**4+ 12.2719 * np.cos(np.radians(tht))**6 -4.0899217 *np.cos(np.radians(tht))**8)/(8.2011129 - 9.6857106 + 13.001805)
         if (GRenddB<0):
             GRenddB=0
         GTenddB = np.cos(np.radians(phi))**2 * GTenddB**0.6
         GRenddB = np.cos(np.radians(phi))**2 * GRenddB**0.65
         GTtdB = GTbroaddB - (7.7827215 * GTenddB - 8.1684971 *GTenddB**2 + 11.627713 * GTenddB**3)
         GRtdB = GRbroaddB - (8.2011129 * GTenddB - 9.6857106 *GTenddB**2 + 13.001805 * GTenddB**3)
     else:
         GTenddB = (-0.021062092 + 7.1497901 *np.cos(np.radians(tht))**2 - 3.3448948 * np.cos(np.radians(tht))**4+ 13.533221 * np.cos(np.radians(tht))**6 - 4.7793303 *np.cos(np.radians(tht))**8)/(6.4776762 - 4.5794876 + 8.1254215)
         if (GTenddB<0):
             GTenddB=0
         GRenddB = (-0.005270753 + 6.3391005 *np.cos(np.radians(tht))**2 + 3.3638038 * np.cos(np.radians(tht))**4+ 0.36978112 * np.cos(np.radians(tht))**6 + 2.3794633 *np.cos(np.radians(tht))**8)/(6.2636256 - 3.9807794 + 7.4747311)
         if (GRenddB<0):
             GRenddB=0
         GTenddB = np.cos(np.radians(phi))**2 * GTenddB**0.6
         GRenddB = np.cos(np.radians(phi))**2 * GRenddB**0.65
         GTtdB = GTbroaddB - (6.4776762 * GTenddB -4.5794876 *GTenddB**2 + 8.1254215 * GTenddB**3)
         GRtdB = GRbroaddB - (6.2636256 * GTenddB - 3.9807794 *GTenddB**2 + 7.4747311 * GTenddB**3)
     GTt = 10**(GTtdB/10)
     GRt = 10**(GRtdB/10)
     Gt = GTt*GRt
     return (GRt)

array_list.append(np.array([], dtype =float))

b = GtGr(array_list[11][0], array_list[12][0],254)


for i in range(len(array_list[11])):
    G7 = GtGr(array_list[11][i],array_list[12][i],254)
    G6 = GtGr(array_list[11][i],array_list[12][i],344)
    array_list[15] = np.append(array_list[15], G7/G6)
for i in range(len(array_list[11])):
    if array_list[15][i]> np.absolute(array_list[15].mean()+3*np.std(array_list[15])):
        for j in range(len(array_list)):
            array_list[j] = np.delete(array_list[j],i)
        

print(array_list[15][:10],array_list[4][:10])

import matplotlib.pyplot as plt
import numpy as np

# Finding the 10 points around the maximum in array_list[11]
max_index = np.argmax(array_list[11])
half_window = 5  # half of 10 points

# Ensure the indices are within bounds
start_index = max(0, max_index - half_window)
end_index = min(len(array_list[11]), max_index + half_window + 1)

# Extract points around the maximum
x_points = array_list[12]
y_points = array_list[11]

# Plotting the points
fig = plt.figure()
ax = fig.add_subplot(projection='polar')

# Fixing the area and color arrays
N = len(x_points)
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta[:N]  # Make sure colors array has the correct length

c = ax.scatter(np.deg2rad(x_points), y_points, c= array_list[15], cmap='Reds', alpha=0.75)
plt.title("Polar Plot of Gain Ratio. Theta is phi in radians, Radius is theta (radar theta) in degrees. Colour is gain ratio (G7/G6)")
plt.show()