import numpy as np
import matplotlib.pyplot as plt
import csv

# Make a random dataset:
height = [188.4, 47.4, 33.3, 27.8,]
bars = ("1","4","8","12")
y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, height)

# Create names on the x-axis
plt.xticks(y_pos, bars)

plt.plot()
plt.title('Hiddenlayer1=128 | hiddenlayer2=128')
plt.xlabel('Thread Count')
plt.ylabel('Time')
plt.savefig('threadtime.png')
