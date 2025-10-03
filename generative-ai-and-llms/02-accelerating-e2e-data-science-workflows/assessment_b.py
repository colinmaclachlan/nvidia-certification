# Step 2: Correct

%load_ext cudf.pandas

import pandas as pd
import cuml
import cupy as cp

df = pd.read_csv('./data/week2.csv', usecols=['lat', 'long', 'infected'])
# df
hospitals_df = pd.read_csv('./data/hospitals.csv')
# hospitals_df
clinics_df = pd.read_csv('./data/clinics.csv')
# clinics_df
all_med = pd.concat([hospitals_df, clinics_df])
# all_med
all_med = all_med.dropna(subset=['Latitude', 'Longitude'])
# all_med
all_med = all_med.reset_index(drop=True)
# all_med

# https://www.ordnancesurvey.co.uk/docs/support/guide-coordinate-systems-great-britain.pdf

def latlong2osgbgrid_cupy(lat, long, input_degrees=True):
    '''
    Converts latitude and longitude (ellipsoidal) coordinates into northing and easting (grid) coordinates, using a Transverse Mercator projection.
    
    Inputs:
    lat: latitude coordinate (N)
    long: longitude coordinate (E)
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

    N0 = -100000 # northing of true origin
    E0 = 400000 # easting of true origin
    F0 = .9996012717 # scale factor on central meridian
    phi0 = 49 * cp.pi / 180 # latitude of true origin
    lambda0 = -2 * cp.pi / 180 # longitude of true origin and central meridian
    
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

cupy_lat = cp.asarray(all_med['Latitude'])
cupy_long = cp.asarray(all_med['Longitude'])

all_med['northing'], all_med['easting'] = latlong2osgbgrid_cupy(cupy_lat, cupy_long)
# all_med

knn = cuml.NearestNeighbors(n_neighbors=1)
knn.fit(all_med[['northing', 'easting']])
# all_med

infected_filter = df['infected']==1
infected_df = df[infected_filter]
infected_df.reset_index(drop=True)
# infected_df

cupy_infected_lat = cp.asarray(infected_df['lat'])
cupy_infected_long = cp.asarray(infected_df['long'])

infected_df['northing'], infected_df['easting'] = latlong2osgbgrid_cupy(cupy_infected_lat, cupy_infected_long)
# infected_df

distances, indices = knn.kneighbors(infected_df[['easting', 'northing']], 1)
infected_df['distance']=distances
infected_df['closest_clinic_hospital']=indices

patients=[
    {
        "name":"John Smith", 
        "northing":462643.2282915547,
        "easting":363859.7580565622
    }, 
    {
        "name":"Greg Brown", 
        "northing":409324.1030472915,
        "easting":464084.5085059871
    }
]

patients_df=pd.DataFrame(patients)

patients_df['distance'], patients_df['closest_clinic_hospital'] = knn.kneighbors(patients_df[['northing', 'easting']])
patients_df['OrganisationID'] = patients_df.apply(lambda x: all_med.loc[x['closest_clinic_hospital'], "OrganisationID"], axis=1)
patients_df.to_json('my_assessment/question_2.json', orient='records')
# patients_df
# all_med