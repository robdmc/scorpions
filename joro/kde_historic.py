import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import rioxarray
from sklearn.neighbors import KernelDensity

# Get matrices/arrays of species IDs and locations
# joroData = pd.read_csv("clavata data for foundations of programming project.csv")
# modifiedJoroData = pd.DataFrame(joroData, columns = ['Year','latitude', 'longitude'])
joroData = pd.read_csv("NA Joro observations 10_19_2022.csv")
modifiedJoroData = pd.DataFrame(joroData, columns = ['observed_on','longitude', 'latitude', 'year'])
year = (pd.DataFrame(joroData, columns = ['year'])["year"]-2014).tolist()

from mpl_toolkits.basemap import Basemap
minLat = modifiedJoroData['latitude'].min()-0.1
minLong = modifiedJoroData['longitude'].min()-0.1
maxLat = modifiedJoroData['latitude'].max()+0.1
maxLong = modifiedJoroData['longitude'].max()+0.1 

#years = ['2014', '2017', '2018', '2019', '2020', '2021', '2022', '2050']
years = ['2050']
cmaps = ['Purples', 'Reds', 'Greens', 'GnBu', 'Blues', 'Oranges', 'Greys']

# Set up the data grid for the contour plot
xgrid = np.arange(minLong, maxLong, 0.05)
ygrid = np.arange(minLat, maxLat, 0.05)
X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy = np.radians(xy)

for i, year in enumerate(years):
    plt.title(years[i])
    
    # plot coastlines with basemap
    m = Basemap(projection='cyl', llcrnrlat=modifiedJoroData['latitude'].min(),
                urcrnrlat=modifiedJoroData['latitude'].max(), llcrnrlon=modifiedJoroData['longitude'].min(),
                urcrnrlon=modifiedJoroData['longitude'].max(), resolution='i')
    m.fillcontinents(color='#FFFFFF', lake_color = 'aqua')
    m.drawcoastlines()
    m.drawstates()
    m.drawcounties(zorder=20)
    
    # construct a spherical kernel density estimate of the distribution
    kde = KernelDensity(bandwidth=0.0035, metric='haversine')
    yearData = modifiedJoroData[modifiedJoroData['year']==int(years[i])]
    kde.fit(np.radians(yearData[['latitude', 'longitude']])) #INVESTIGATE YEAR
    
    # evaluate only on the land: -9999 indicates ocean
    Z = np.full(xy.shape[0], 0)
    Z = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    m.scatter(yearData['longitude'], yearData['latitude'], s=2, c = 'blue', zorder=3, cmap='black', latlon=True)

    # plot contours of the density
    levels = np.linspace(300, Z.max(), 25)
    cp = plt.contourf(X, Y, Z, levels=levels, cmap='Reds')
    fig=plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.colorbar(cp)
    plt.show()
    