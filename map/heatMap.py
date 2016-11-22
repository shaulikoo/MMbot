import time

import matplotlib.pyplot as plt

from algorithm.map.point import *



#GENERATE the map (3 - points)
start = time.time()
map = [[0 for x in range(5)] for x in range(10)]
for k in range(0, 10):
    for l in range(0, 5):
        map[k][l] = MapPoint()
        map[k][l].x = k
        map[k][l].y = l
        map[k][l].data = np.random.rand(100)*100
makeMap = time.time()
print "Map generator running time: " + str(makeMap-start)

#create checkPoint
checkPoint = ScanPoint()
checkPoint.data = np.random.rand(100)*100

# create the figure
fig = plt.figure()
ax = fig.add_subplot(111)
sidex = np.linspace(0, 10, 200)
sidey = np.linspace(0, 5, 200)
X, Y = np.meshgrid(sidex, sidey)
plt.show(block=False)
for loc in range(0, 10):
    Z = 0
    checkPoint.data[2] += loc*5
    for i in range(0, len(map)):
        for j in range(0, len(map[1])):
            #Locate
            Z += checkPoint.creGaussian(map[i][j], X, Y)
    im = ax.pcolormesh(X, Y, Z)
    fig.canvas.draw()
checkVectorTime = time.time()
print "check Vector for other vectors time: " + str(checkVectorTime-makeMap)
plt.show(block=True)


'''
# draw some data in loop
for i in range(100):
    # wait for a second
    time.sleep(0.01)
    Z = np.exp(-(X+2) - (Y-2))
    im = ax.pcolormesh(X,Y,Z)
    # redraw the figure
    fig.canvas.draw()
'''