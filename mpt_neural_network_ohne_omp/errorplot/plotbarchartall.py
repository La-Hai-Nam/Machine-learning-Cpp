import numpy as np
import matplotlib.pyplot as plt
import csv

# Make a random dataset:
height = [189, 29, 30, 8.1, 28, 7.8, 22, 7.9, 27, 8, 25, 7.1]
bars = ("CFLAGS1", "CFLAGS1 omp", "CFLAGS2", "CFLAGS2 omp", "CFLAGS3", "CFLAGS3 omp", "CFLAGS4","CFLAGS4 omp", "CFLAGS5", "CFLAGS5 omp", "CFLAGS6", "CFLAGS6 omp")
y_pos = np.arange(len(bars))

# Create bars
plt.figure(figsize=(15, 3))
plt.bar(y_pos, height,width=0.8,color=["blue", "red", "blue", "red", "blue", "red"])

# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.plot()
plt.title('Hiddenlayer1 = 128 | hiddenlayer2 = 128')
plt.xlabel('Thread Count')
plt.ylabel('Time')
plt.subplots_adjust(bottom=0.15)
plt.savefig('threadtimeoptimize.png')
