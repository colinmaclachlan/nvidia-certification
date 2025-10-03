import pandas as pd

import numpy as np
import cupy as cp
import time

# DO NOT CHANGE THIS CELL
num_ele = 1000000

df = pd.DataFrame(
    {
        "a": range(num_ele),
        "b": range(10, num_ele + 10),
        "c": range(100, num_ele + 100),
        "d": range(1000, num_ele + 1000)
    }
)

# preview
df.head()

# DO NOT CHANGE THIS CELL
arr=df.values
# alternative approach
# arr=cp.asarray(df)

start=time.time()

display(arr.sum(axis=1))

time.time()-start

df['sum']=arr.sum(axis=1)

df.head()

dtype_dict={
    'age': 'int8', 
    'sex': 'category', 
    'county': 'category', 
    'lat': 'float32', 
    'long': 'float32', 
    'name': 'category'
}
        
df=pd.read_csv('./data/uk_pop.csv', dtype=dtype_dict)
df.head()

# https://www.ordnancesurvey.co.uk/docs/support/guide-coordinate-systems-great-britain.pdf

def latlong2osgbgrid(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (north)
    long: longitude coordinate (east)
    input_degrees: if True (default), interprets the coordinates as degrees; otherwise, interprets coordinates as radians
    
    Output:
    (northing, easting)
    '''
    
    if input_degrees:
        lat = lat * np.pi/180
        long = long * np.pi/180

    a = 6377563.396
    b = 6356256.909
    e2 = (a**2 - b**2) / a**2

    N0 = -100000                # northing of true origin
    E0 = 400000                 # easting of true origin
    F0 = .9996012717            # scale factor on central meridian
    phi0 = 49 * np.pi / 180     # latitude of true origin
    lambda0 = -2 * np.pi / 180  # longitude of true origin and central meridian
    
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    tanlat = np.tan(lat)
    
    latdiff = lat-phi0
    longdiff = long-lambda0

    n = (a-b) / (a+b)
    nu = a * F0 * (1 - e2 * sinlat ** 2) ** -.5
    rho = a * F0 * (1 - e2) * (1 - e2 * sinlat ** 2) ** -1.5
    eta2 = nu / rho - 1
    M = b * F0 * ((1 + n + 5/4 * (n**2 + n**3)) * latdiff - 
                  (3*(n+n**2) + 21/8 * n**3) * np.sin(latdiff) * np.cos(lat+phi0) +
                  15/8 * (n**2 + n**3) * np.sin(2*(latdiff)) * np.cos(2*(lat+phi0)) - 
                  35/24 * n**3 * np.sin(3*(latdiff)) * np.cos(3*(lat+phi0)))
    I = M + N0
    II = nu/2 * sinlat * coslat
    III = nu/24 * sinlat * coslat ** 3 * (5 - tanlat ** 2 + 9 * eta2)
    IIIA = nu/720 * sinlat * coslat ** 5 * (61-58 * tanlat**2 + tanlat**4)
    IV = nu * coslat
    V = nu / 6 * coslat**3 * (nu/rho - np.tan(lat)**2)
    VI = nu / 120 * coslat ** 5 * (5 - 18 * tanlat**2 + tanlat**4 + 14 * eta2 - 58 * tanlat**2 * eta2)

    northing = I + II * longdiff**2 + III * longdiff**4 + IIIA * longdiff**6
    easting = E0 + IV * longdiff + V * longdiff**3 + VI * longdiff**5

    return(northing, easting)

# https://www.ordnancesurvey.co.uk/docs/support/guide-coordinate-systems-great-britain.pdf

def latlong2osgbgrid_cupy(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (north)
    long: longitude coordinate (east)
    input_degrees: if True (default), interprets the coordinates as degrees; otherwise, interprets coordinates as radians
    
    Output:
    (northing, easting)
    '''
    
    if input_degrees:
        lat = lat * cp.pi/180
        long = long * cp.pi/180

    a = 6377563.396
    b = 6356256.909
    e2 = (a**2 - b**2) / a**2

    N0 = -100000                 # northing of true origin
    E0 = 400000                  # easting of true origin
    F0 = .9996012717             # scale factor on central meridian
    phi0 = 49 * cp.pi / 180      # latitude of true origin
    lambda0 = -2 * cp.pi / 180   # longitude of true origin and central meridian
    
    sinlat = cp.sin(lat)
    coslat = cp.cos(lat)
    tanlat = cp.tan(lat)
    
    latdiff = lat-phi0
    longdiff = long-lambda0

    n = (a-b) / (a+b)
    nu = a * F0 * (1 - e2 * sinlat ** 2) ** -.5
    rho = a * F0 * (1 - e2) * (1 - e2 * sinlat ** 2) ** -1.5
    eta2 = nu / rho - 1
    M = b * F0 * ((1 + n + 5/4 * (n**2 + n**3)) * latdiff - 
                  (3*(n+n**2) + 21/8 * n**3) * cp.sin(latdiff) * cp.cos(lat+phi0) +
                  15/8 * (n**2 + n**3) * cp.sin(2*(latdiff)) * cp.cos(2*(lat+phi0)) - 
                  35/24 * n**3 * cp.sin(3*(latdiff)) * cp.cos(3*(lat+phi0)))
    I = M + N0
    II = nu/2 * sinlat * coslat
    III = nu/24 * sinlat * coslat ** 3 * (5 - tanlat ** 2 + 9 * eta2)
    IIIA = nu/720 * sinlat * coslat ** 5 * (61-58 * tanlat**2 + tanlat**4)
    IV = nu * coslat
    V = nu / 6 * coslat**3 * (nu/rho - cp.tan(lat)**2)
    VI = nu / 120 * coslat ** 5 * (5 - 18 * tanlat**2 + tanlat**4 + 14 * eta2 - 58 * tanlat**2 * eta2)

    northing = I + II * longdiff**2 + III * longdiff**4 + IIIA * longdiff**6
    easting = E0 + IV * longdiff + V * longdiff**3 + VI * longdiff**5

    return(northing, easting)

# https://www.ordnancesurvey.co.uk/docs/support/guide-coordinate-systems-great-britain.pdf

def latlong2osgbgrid_cupy(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (north)
    long: longitude coordinate (east)
    input_degrees: if True (default), interprets the coordinates as degrees; otherwise, interprets coordinates as radians
    
    Output:
    (northing, easting)
    '''
    
    if input_degrees:
        lat = lat * cp.pi/180
        long = long * cp.pi/180

    a = 6377563.396
    b = 6356256.909
    e2 = (a**2 - b**2) / a**2

    N0 = -100000                 # northing of true origin
    E0 = 400000                  # easting of true origin
    F0 = .9996012717             # scale factor on central meridian
    phi0 = 49 * cp.pi / 180      # latitude of true origin
    lambda0 = -2 * cp.pi / 180   # longitude of true origin and central meridian
    
    sinlat = cp.sin(lat)
    coslat = cp.cos(lat)
    tanlat = cp.tan(lat)
    
    latdiff = lat-phi0
    longdiff = long-lambda0

    n = (a-b) / (a+b)
    nu = a * F0 * (1 - e2 * sinlat ** 2) ** -.5
    rho = a * F0 * (1 - e2) * (1 - e2 * sinlat ** 2) ** -1.5
    eta2 = nu / rho - 1
    M = b * F0 * ((1 + n + 5/4 * (n**2 + n**3)) * latdiff - 
                  (3*(n+n**2) + 21/8 * n**3) * cp.sin(latdiff) * cp.cos(lat+phi0) +
                  15/8 * (n**2 + n**3) * cp.sin(2*(latdiff)) * cp.cos(2*(lat+phi0)) - 
                  35/24 * n**3 * cp.sin(3*(latdiff)) * cp.cos(3*(lat+phi0)))
    I = M + N0
    II = nu/2 * sinlat * coslat
    III = nu/24 * sinlat * coslat ** 3 * (5 - tanlat ** 2 + 9 * eta2)
    IIIA = nu/720 * sinlat * coslat ** 5 * (61-58 * tanlat**2 + tanlat**4)
    IV = nu * coslat
    V = nu / 6 * coslat**3 * (nu/rho - cp.tan(lat)**2)
    VI = nu / 120 * coslat ** 5 * (5 - 18 * tanlat**2 + tanlat**4 + 14 * eta2 - 58 * tanlat**2 * eta2)

    northing = I + II * longdiff**2 + III * longdiff**4 + IIIA * longdiff**6
    easting = E0 + IV * longdiff + V * longdiff**3 + VI * longdiff**5

    return(northing, easting)

numpy_lat = np.asarray(df['lat'])
numpy_long = np.asarray(df['long'])

n_numpy_array, e_numpy_array = latlong2osgbgrid(numpy_lat, numpy_long)

# Exercise #1 - Adding Grid Coordinate Columns to DataFrame

cupy_lat = cp.asarray(df['lat'])
cupy_long = cp.asarray(df['long'])

n_cupy_array, e_cupy_array = latlong2osgbgrid_cupy(cupy_lat, cupy_long)
df['northing'] = pd.Series(n_cupy_array).astype('float32')
df['easting'] = e_cupy_array.astype('float32')
print(df.dtypes)
df.head()

start=time.time()
display(df.loc[np.logical_and(df['name'].str.startswith('E'), df['name'].str.endswith('D'))].head())
print(f'Duration: {round(time.time()-start, 2)} seconds')

start=time.time()
display(df.loc[cp.logical_and(df['name'].str.startswith('E'), df['name'].str.endswith('D'))].head())
print(f'Duration: {round(time.time()-start, 2)} seconds')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)