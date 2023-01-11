import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__)
import cufflinks as cf

# For Notebooks
init_notebook_mode(connected=True)

# For offline use
cf.go_offline()

df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df.head()

df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2.head()

#Using Cufflinks and iplot()
df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)

#Bar Plots
df2.iplot(kind='bar',x='Category',y='Values')
df.count().iplot(kind='bar')

#Boxplots
df.iplot(kind='box')

#3d Surface
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3.iplot(kind='surface',colorscale='rdylbu')

#Spread
df[['A','B']].iplot(kind='spread')

#histogram
df['A'].iplot(kind='hist',bins=25)
df.iplot(kind='bubble',x='A',y='B',size='C')

#scatter_matrix()
df.scatter_matrix()


