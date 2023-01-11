import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
get_ipython().run_line_magic('matplotlib', 'inline')

df3.info()
df3.head()

#Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure
#out how to stretch it in a similar fashion.
df3.plot.scatter(x='a',y='b',c='red',s=50,figsize=(12,3))

#Create a histogram of the 'a' column.
df3['a'].plot.hist()

#These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' 
#and redo the histogram from above. Also figure out how to add more bins to it.
plt.style.use('ggplot')
df3['a'].plot.hist(alpha=0.5,bins=25)

#Create a boxplot comparing the a and b columns.
df3[['a','b']].plot.box()

#Create a kde plot of the 'd' column 
df3['d'].plot.kde()

#Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**
df3['d'].plot.density(lw=5,ls='--')

#Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).
df3.ix[0:30].plot.area(alpha=0.4)

#Notice how the legend in our previous figure overlapped some of actual diagram. Can you figure out how to display the legend outside of the plot as shown below?**
f = plt.figure()
df3.ix[0:30].plot.area(alpha=0.4,ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()



