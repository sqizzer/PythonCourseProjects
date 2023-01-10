import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.arange(0,100)
y = x*2
z = x**2


#Exercise 1
#Create a figure object called fig using plt.figure()
#Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax.
#Plot (x,y) on that axes and set the labels and titles to match the plot below:

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')


#Exercise 2
#Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.

fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])

#Now plot (x,y) on both axes. And call your figure object to show it.
ax1.plot(x,y)
ax1.set_xlabel('x')
ax1.set_ylabel('y')


ax2.plot(x,y)
ax2.set_xlabel('x')
ax2.set_ylabel('y')

fig # Show figure object

#Exercise 3
#Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.4,.4])

#Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:
ax.plot(x,z)
ax.set_xlabel('X')
ax.set_ylabel('Z')

ax2.plot(x,y)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('zoom')
ax2.set_xlim(20,22)
ax2.set_ylim(30,50)

fig
#Exercise 4
#Use plt.subplots(nrows=1, ncols=2) to create the plot below.
#Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)

#Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style
axes[0].plot(x,y,color="blue", lw=3, ls='--')
axes[1].plot(x,z,color="red", lw=3, ls='-')
fig

#See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code.
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,2))
axes[0].plot(x,y,color="blue", lw=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].plot(x,z,color="red", lw=3, ls='--')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')



