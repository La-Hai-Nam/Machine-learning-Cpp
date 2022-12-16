# import matplotlib.pyplot as plt
# import csv
 
# X = []
# Y = []
# X1 = []
# Y1 = []
# with open("errorplot.txt", 'r') as datafile:
#     plotting = csv.reader(datafile, delimiter=',')

#     for ROWS in plotting:
#         X.append(float(ROWS[0]))
#         Y.append(float(ROWS[1]))
 
# with open("errorplottan.txt", 'r') as datafile:
#     plotting2 = csv.reader(datafile, delimiter=',')
#     for ROWS in plotting2:
#         X1.append(float(ROWS[0]))
#         Y1.append(float(ROWS[1]))


# plt.plot(X, Y,label = 'sigmoid')
# plt.plot(X, Y1,label = 'tanh')
# plt.title('Line Graph')
# plt.xlabel('Epoch')
# plt.ylabel('Net Recent Error')
# plt.legend()
# plt.savefig('errorcomb.png')
import matplotlib.pyplot as plt
import csv
 
X = []
Y = []

with open("errorplottan.txt", 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')

    for ROWS in plotting:
        X.append(float(ROWS[0]))
        Y.append(float(ROWS[1]))
 


plt.plot(X, Y,label = 'sigmoid')
plt.title('Line Graph')
plt.xlabel('Epoch')
plt.ylabel('Net Recent Error')
plt.legend()
plt.savefig('errorcomb.png')
