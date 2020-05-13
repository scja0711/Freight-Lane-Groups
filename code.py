def clear_folder(folder):
    '''Deletes all items in a folder'''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def rgb_hex(rgb):
    '''converts rgb to hex'''
    rgb = eval(rgb)
    rgb = [str(hex(int(x*255)))[2:].capitalize() for x in rgb]
    rgb = ['00' if s=='0' else s for s in rgb]
    return '#' + ''.join(rgb)

def cities_get():
    '''Retrieves city, state, population, and coordinates, of the 1000 largest US cities.'''
    url = "https://public.opendatasoft.com/explore/dataset/1000-largest-us-cities-by-population-with-geographic-coordinates/download/?format=csv&timezone=America/Chicago&use_labels_for_header=true&csv_separator=%3B"
    columns = ['City', 'State', 'Population', 'Coordinates']
    df = pd.read_csv(url, sep=';', usecols=columns)
    df.rename(columns=dict([(c, c.lower()) for c in df.columns]), inplace=True)
    df = df.loc[~df['state'].isin(['Hawaii', 'Alaska']), :]
    df['lon'], df['lat'] = df['coordinates'].str.split(',', 1).str
    df[['lat', 'lon']] = df[['lat', 'lon']].astype(float)
    df.reset_index(drop=True, inplace=True)
    del df['coordinates']
    return df
    
def cities_projection(df, proj_params):
    '''Transforms a DataFrame to GeoDataFrame using an equidistant conic projection.
    The standard parallel, or latitude of the y origin, is 37 degrees north.
    The center vertical, or longitude of the x origin, is -96 degrees east.'''
    crs = {'init':'epsg:4326'}
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lat'], df['lon']), crs=crs)
    df = df.to_crs(proj_params)
    proj = Proj(df.crs)
    df['x'] = df['geometry'].x
    df['y'] = df['geometry'].y
    df['geometry'] = gpd.points_from_xy(x=df['x'], y=df['y'])
    return df, proj

def routes_histogram(distance, min_length):
    '''Shows distribution of route lengths'''
    x = np.triu(distance).flatten()
    x = x[x!=0]
    handles_list = []
    bins = np.linspace(0, 5000, 11)
    mask = x >= min_length
    weights = np.ones_like(x) / float(len(x))
    color = (254,254,98), (211,95,183)
    color = [tuple((a/255 for a in rgb)) for rgb in color]
    kwargs = {'bins': bins, 'weights': weights, 'stacked':True, 'alpha':1}
    kwargs['color'], kwargs['weights'] = color[0], weights
    heights, bins, handles = plt.hist(x, **kwargs)
    handles_list.append(handles[0])
    kwargs['color'], kwargs['weights'] = color[1], weights[mask]
    heights, bins, handles = plt.hist(x[mask], **kwargs)
    handles_list.append(handles[0])
    plt.xticks(bins.tolist()[::2])
    plt.title('Histogram of all possible routes by length', color=(1,1,1))
    labels = [fmt.format(min_length) for fmt in ('Shorter than {} km', '{} km or longer')]
    plt.legend(handles=handles_list, labels=labels)
    plt.xlabel('Route length (km)')
    plt.ylabel('Proportion of routes')
    plt.savefig(directory + '/_route_lengths.png')
    plt.close()
    return

def routes_define(df, distance, min_distance, total_drawn, endpoint_stdev):
    '''Draws n routes between cities selected at random.'''
    df, proj = df.copy(), Proj(df.crs)
    available = np.argwhere(distance >= min_distance)
    ava = pd.DataFrame(available, columns=['orig', 'dest'], index=range(len(available)))
    pop = df['population'].rank(pct=True).to_frame('pop')
    scaler = MinMaxScaler((0.5, 1.0))
    pop = dict(zip(pop.index, scaler.fit_transform(pop.values).flatten().tolist()))
    ava[['orig_proba', 'dest_proba']] = ava.applymap(lambda x: pop[x])
    ava['route_proba'] = ava['orig_proba'] * ava['dest_proba']
    drawn = ava.sample(n=total_drawn, replace=True, random_state=1, weights='route_proba')
    drawn_cities = drawn['orig'].values.tolist() + drawn['dest'].values.tolist()
    df = df.loc[drawn_cities, :]
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'city_index'}, inplace=True)
    df[['x', 'y']] += np.random.randn(total_drawn*2, 2) * endpoint_stdev
    df['geometry'] = gpd.points_from_xy(x=df['x'], y=df['y'])
    df['route'] = list(range(total_drawn))*2
    df['endpoint'] = ['orig']*total_drawn + ['dest']*total_drawn
    df['coords'] = df.apply(lambda row: list(proj(row['x'], row['y'], inverse=True)), axis=1)
    df[['lon', 'lat']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    return df, len(available)
    
def routes_cluster(df, orig, dest, params, color, size):
    '''Groups routes using the OPTICS clustering algorithm.'''
    model = OPTICS(**params)
    X = np.hstack((df.loc[orig, ['x','y']].values, df.loc[dest, ['x','y']].values))
    model.fit(X=X)
    n = len(X)
    fancy = model.ordering_.tolist()
    fancy2 = fancy + [f + n for f in model.ordering_]
    df = df.loc[fancy2, :]
    df['reachability_order'] = list(range(n)) * 2
    df['reachability'] = model.reachability_[fancy].tolist() * 2
    df['route_cluster'] = model.labels_[fancy].tolist() * 2
    df['route_cluster'] = df['route_cluster'].replace(-1, np.nan)
    df['reachability_plot'] = df['route_cluster'] // size
    df.loc[orig, 'reachability_plot'] = df.loc[orig, 'reachability_plot'].fillna(method='bfill')
    df.loc[dest, 'reachability_plot'] = df.loc[dest, 'reachability_plot'].fillna(method='bfill')
    df.loc[orig, 'reachability_plot'] = df.loc[orig, 'reachability_plot'].fillna(method='ffill')
    df.loc[dest, 'reachability_plot'] = df.loc[dest, 'reachability_plot'].fillna(method='ffill')
    df['reachability_color'] = (df['route_cluster'] % size).map(color)
    df['reachability_color'] = df['reachability_color'].fillna('(1,1,1)')
    return df

def routes_reachability(df, file_number):
    '''Plots the reachability of routes as defined by OPTICS.'''
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    first, last = int(df['route_cluster'].min()), int(df['route_cluster'].max())
    title = 'Reachability Plot of Route Clusters {} - {}'.format(first, last)
    ax.set_title(title, fontdict={'fontsize': 20})
    ax.set_ylabel('Reachability Distance', fontsize=16)
    ax.set_xlabel('Route', fontsize=16)
    ax.set_facecolor('xkcd:black')
    ax.tick_params(axis='both', which='major', labelsize=16)
    color = [eval(c) for c in df['reachability_color']]
    ax.scatter(x=df['reachability_order'], y=df['reachability'], s=50, c=color)
    plt.savefig(directory + '/_reachability_{}.png'.format(int(file_number)))
    plt.close(fig)
    return

def clusters_projection(df, cluster_params, color, proj):
    '''Creates geometric representations of clusters.'''
    columns = ['route_cluster', 'endpoint', 'geometry', 'x', 'y']
    df = df[columns].dissolve(by=['route_cluster', 'endpoint'], aggfunc='mean')
    df['coords'] = df.apply(lambda row: list(proj(row['x'], row['y'], inverse=True)), axis=1)
    df[['lon','lat']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    df['hull'] = df['geometry'].apply(ConvexHull)
    df['hull_points'] = df['hull'].apply(lambda hull: str(hull.points.tolist()))
    df['hull_coords'] = df['hull_points'].apply(
        lambda pt: str([proj(p[0], p[1], inverse=True) for p in eval(pt)]))
    df['hull_coords']
    mask = df.index.get_level_values(level=0) < 0
    df['map_number'] = df.index.get_level_values(level=0) // 5
    df.loc[mask, 'map_number'] = -1
    df['cluster_color'] = df.index.get_level_values(level=0) % 5
    df['cluster_color'] = df['cluster_color'].map(color)
    df['cluster_color'] = df['cluster_color'].fillna('#FFFFFF')
    return df

def clusters_partition(df, color, size):
    params = {'cluster_method':'xi', 'metric':'cityblock', 'xi':0.05,
              'min_cluster_size':None, 'max_eps':np.inf, 'n_jobs':None}
    model = OPTICS(**params)
    df['hub'] = model.fit_predict(X=df[['x','y']].values)
    mask = df['hub'] < 0
    df['hub_group'] = df['hub'] // size
    df.loc[mask, 'hub_group'] = -1
    df['hub_color'] = df['hub'] % size
    df.loc[mask, 'hub_color'] = -1
    df['hub_color'] = df['hub_color'].map(color)
    return df

def hub_page(clusters, routes):
    clusters.reset_index(inplace=True)
    clusters.set_index(['route_cluster', 'endpoint'], inplace=True)
    routes.reset_index(inplace=True)
    routes.set_index(['route_cluster', 'endpoint'], inplace=True)
    mask = clusters['hub'] != -1
    page = clusters.loc[mask, ['hub']].copy()
    page = pd.merge(page, routes[['route', 'x', 'y']], on=page.index.names, how='inner')
    counts = page['route'].value_counts()
    route_list = counts.index[counts == 2]
    mask = page['route'].isin(route_list)
    page = page.loc[mask, ['route', 'hub', 'x', 'y']]
    return page, clusters, routes

def hub_rank(page):
    page.reset_index(inplace=True)
    page.set_index(['route', 'endpoint'], inplace=True)
    page.sort_index(ascending=True, inplace=True)
    page['route_cluster'] = page['route_cluster'].astype(int)
    orig, dest = page.loc[idx[:,'orig'], ['hub']].values, page.loc[idx[:,'dest'], ['hub']].values
    graph = np.hstack((orig, dest))
    weights = np.ones(graph.shape[0])
    nodes = np.unique(graph.flatten())
    shape = [len(nodes)] * 2
    position_tuple = graph[:,0], graph[:,1]
    graph_sparse = sparse.csr_matrix((weights, position_tuple), shape=shape)
    graph = pd.DataFrame(graph, columns=['orig', 'dest'])
    page_rank = pagerank(graph_sparse, p=0.85)
    rank_dict = dict(zip(nodes, page_rank))
    page['rank'] = page['hub'].map(rank_dict)
    rank = page.groupby('hub')['rank'].first().to_frame('rank')
    return page, rank, graph

def rank_format(df, page, clusters, color_hex):
    hub_color = clusters.copy().reset_index()
    hub_color = hub_color[['hub', 'hub_color']].drop_duplicates()    
    hub_color.set_index('hub', inplace=True)
    hub_color = hub_color['hub_color']
    scaler = MinMaxScaler((0.1, 1.0))
    df['scaled'] = scaler.fit_transform(df[['rank']].values)
    df['place'] = df['rank'].rank(method='dense', ascending=False).astype(int) - 1
    mask = df['place'] > 4
    df.loc[mask, 'place'] = -1
    df['color'] = df['place'].map(color_hex)
    df[['x', 'y']] = page.groupby('hub')[['x', 'y']].mean()
    df['coords'] = df.apply(lambda row: list(proj(row['x'], row['y'], inverse=True)), axis=1)
    df[['lon', 'lat']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    mask = df['place'] == -1
    df['alpha'] = 1
    df.loc[mask, 'alpha'] = 0.2
    scaler = MinMaxScaler()
    return df

def edge_define(graph):
    scaler = MinMaxScaler((0.1, 1.0))
    df = graph.groupby(['orig', 'dest']).size()
    df.sort_values(ascending=False, inplace=True)
    df = df.to_frame('route_count')
    df['scaled'] = scaler.fit_transform(df[['route_count']].values)
    df['x_orig'] = df.index.get_level_values('orig').map(rank['x'])
    df['y_orig'] = df.index.get_level_values('orig').map(rank['y'])
    df['x_dest'] = df.index.get_level_values('dest').map(rank['x'])
    df['y_dest'] = df.index.get_level_values('dest').map(rank['y'])
    return df

def edge_triangle(df):
    df['dx'] = df['x_orig'] - df['x_dest']
    df['dy'] = df['y_orig'] - df['y_dest']
    df['length'] = (df['dx']**2 + df['dy']**2).apply(np.sqrt)
    df['dx'] /= df['length']
    df['dy'] /= df['length']
    df['x_top'] = df['x_dest'] - df['scaled'] * df['dy'] * 100
    df['y_top'] = df['y_dest'] + df['scaled'] * df['dx'] * 100
    df['x_bottom'] = df['x_dest'] + df['scaled'] * df['dy'] * 100
    df['y_bottom'] = df['y_dest'] - df['scaled'] * df['dx'] * 100
    return df

def edge_format(df):
    post_list = ['_orig', '_dest', '_top', '_bottom']
    for post in post_list:
        x, y, lon, lat = [s + post for s in ['x', 'y', 'lon', 'lat']]
        df['coords'] = df.apply(lambda row: list(proj(row[x], row[y], inverse=True)), axis=1)
        df[[lon, lat]] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    return df

def clusters_geometric(df, hull_maps):
    df.reset_index(inplace=True)
    df.set_index(['map_number', 'route_cluster', 'endpoint'], inplace=True)
    map_unique = df.index.get_level_values(level='map_number').unique()
    for map_number in map_unique[:hull_maps]:
        map_df = df.loc[(map_number), :]
        cluster_unique = map_df.index.get_level_values(level='route_cluster').unique()
        m = folium.Map(location=[37,-96], tiles='CartoDB dark_matter', zoom_start=4)
        for route_cluster in cluster_unique:
            route_df = df.loc[(map_number, route_cluster), :]
            endpoint_list = route_df.index.get_level_values(level='endpoint')
            for endpoint in endpoint_list:
                i = (map_number, route_cluster, endpoint)
                hull = np.array(eval(df.loc[i, 'hull_coords']))
                vertices = df.loc[i, 'hull'].vertices.tolist()
                locations = hull[vertices][:,[1,0]]
                color = df.loc[i, 'cluster_color']
                folium.Polygon(locations=locations, color=color, weight=4, fill=True,
                               fill_color=color, fill_opacity=0.1).add_to(m)
        m.save(directory + '/_hull_{}.html'.format(int(map_number)))
    return

def routes_show(df, routes_per):
    df.index.name = 'point_id'
    df.reset_index(inplace=True)
    df.set_index(['route', 'endpoint'], inplace=True)
    df.sort_index(ascending=False, inplace=True)
    route_list = df.index.get_level_values(level=0).unique()
    m = folium.Map(location=[37,-96], tiles='CartoDB dark_matter', zoom_start=4)
    for route_id in route_list[:routes_per]:
        orig, dest = df.loc[route_id, ['lat', 'lon']].values.tolist()
        folium.Circle(location=orig, radius=15*(10**3), color='#FFFFFF', fill=True,
                      fill_opacity=0.5, weight=1).add_to(m)
        folium.Circle(location=dest, radius=15*(10**3), color='#FFFFFF', fill=True,
                      fill_opacity=0.5, weight=1).add_to(m)
        folium.PolyLine(locations=[orig, dest], color='#FFFFFF', weight=1, opacity=0.2).add_to(m)
    m.save(directory + '/a_route.html')
    return

def clusters_monochromatic(df):
    df.reset_index(inplace=True)
    df.set_index(['route_cluster', 'endpoint'], inplace=True)
    df.sort_index(ascending=False, inplace=True)
    clusters_list = df.index.get_level_values(level='route_cluster').unique()
    m = folium.Map(location=[37,-96], tiles='CartoDB dark_matter', zoom_start=4)
    for route_cluster in clusters_list:
        orig, dest = df.loc[(route_cluster), ['lat', 'lon']].values.tolist()
        folium.Circle(location=orig, radius=15*(10**3), color='#FFFFFF', fill=True,
                      fill_opacity=0.5, weight=1).add_to(m)
        folium.Circle(location=dest, radius=15*(10**3), color='#FFFFFF', fill=True,
                      fill_opacity=0.5, weight=1).add_to(m)
        folium.PolyLine(locations=[orig, dest], color='#FFFFFF', weight=1, opacity=0.2).add_to(m)
    m.save(directory + '/b_route_clusters.html')
    return len(clusters_list)

def clusters_polychromatic(df):
    df.reset_index(inplace=True)
    df.set_index(['hub_group', 'route_cluster', 'endpoint'], inplace=True)
    df.sort_index(ascending=False, inplace=True)
    swap_endpoint = {'dest': 'orig', 'orig': 'dest'}
    i = idx[:, :, :]
    level_up = 'hub_group'
    for hub_group in df.loc[i, :].index.get_level_values(level=level_up).unique():
        m = folium.Map(location=[37,-96], tiles='CartoDB dark_matter', zoom_start=4)
        i = idx[hub_group, :, :]
        level_up = 'route_cluster'
        for route_cluster in df.loc[i, :].index.get_level_values(level=level_up).unique():
            i = idx[hub_group, route_cluster, :]
            level_up = 'endpoint'
            for endpoint in df.loc[i, :].index.get_level_values(level=level_up).unique():
                i_1 = idx[hub_group, route_cluster, endpoint]
                endpoint = swap_endpoint[endpoint]
                i_2 = idx[        :, route_cluster, endpoint]
                color = df.loc[i_1, 'hub_color']
                location_1 = df.loc[i_1, ['lat', 'lon']]
                location_2 = df.loc[i_2, ['lat', 'lon']]
                for location in [location_1, location_2]:
                    folium.Circle(location=location, radius=15*10**3, color=color, fill=True,
                                  fill_opacity=0.5, weight=1).add_to(m)
                folium.PolyLine(locations=[location_1, location_2], color=color, weight=1,
                                opacity=0.5).add_to(m)
        m.save(directory + '/c_clusters_polychromatic_{}.html'.format(hub_group))
    return

def hub_show(clusters, rank, edge):
    mask = rank['place'] != -1
    rank_index = rank.index[mask]
    mask = edge.index.get_level_values('dest').isin(rank_index)
    m = folium.Map(location=[37,-96], tiles='CartoDB dark_matter', zoom_start=4)
    for cluster in edge.index[mask]:
        weight = edge.loc[cluster, 'scaled'] * 5
        color = rank.loc[cluster[1], 'color']
        alpha = rank.loc[cluster[1], 'alpha']
        orig = edge.loc[cluster, ['lat_orig', 'lon_orig']].tolist()
        dest = edge.loc[cluster, ['lat_dest', 'lon_dest']].tolist()
        dest_top = edge.loc[cluster, ['lat_top', 'lon_top']].tolist()
        dest_bottom = edge.loc[cluster, ['lat_bottom', 'lon_bottom']].tolist()
        folium.Polygon(locations=[orig, dest_top, dest_bottom], color=color, weight=1,
                       fill=True, fill_color=color, fill_opacity=0.2, opacity=alpha).add_to(m)
    for hub in rank.index:
        location = rank.loc[hub, ['lat', 'lon']].values.tolist()
        radius = rank.loc[hub, 'rank'] * 10**6
        color = rank.loc[hub, 'color']
        alpha = rank.loc[hub, 'alpha']
        weight = rank.loc[hub, 'scaled'] * 5
        folium.Circle(location=location, weight=1.5, radius=radius, fill=True, color=color,
                      fill_opacity=0.2).add_to(m)
        folium.Circle(location=location, weight=1.5, radius=radius/2, fill=True, color=color,
                      fill_opacity=1).add_to(m)
    m.save(directory + '/e_hubs.html')
    return


import os
import gc
import time
import shutil
import folium
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib import cm
from pyproj import Proj, transform
from fast_pagerank import pagerank
from scipy import sparse
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

# Parameters for OPTICS clustering
cluster_params       = {'cluster_method':'xi', 'metric':'cityblock', 'xi':0.05,
                        'min_cluster_size':None, 'n_jobs':None, 'max_eps':np.inf}

# Parameters for the map projection
proj_params          = {'proj':'eqdc', 'datum':'NAD83', 'units':'km', 'x_0':0, 'y_0':0,
                        'lat_0': 37, 'lat_1': 37, 'lat_2': 37,
                        'lon_0':-96, 'lon_1':-96, 'lon_2':-96}

cities_total         = 200     # Total cities to choose from; the max is 998
routes_total         = 5000    # Total routes to create for clustering
min_distance         = 650     # Minimum route length
endpoint_stdev       = 200     # Standard deviation of endpoint coordinates (x, y)

reachability_plots   = 5       # Total reachability plots
hull_maps            = 5       # Total hull maps

reachabilities_per   = 5       # Routes per reachability plot
routes_per           = 200     # Routes per map
hubs_per             = 5       # Hubs per map

# delete old files
directory = 'D:\portfolio\clustering\outputs'
clear_folder(directory)
gc.collect()

# color
set_1 = [[c*255 for c in color] for color in plt.cm.Set1.colors]
set_1 = set_1[:6]
color_list = set_1
color_rgb = {i: str([c/255 for c in color]) for i, color in enumerate(color_list)}
color_rgb[-1] = '[1,1,1]'
color_hex = {i: rgb_hex(c) for i, c in color_rgb.items()}

# cities
if 'cities_raw' not in dir():
    cities_raw = cities_get()
else:
    pass
cities = cities_raw.copy()
total_cities = cities.shape[0]
cities = cities.sample(cities_total)
cities.reset_index(drop=True, inplace=True)
cities, proj = cities_projection(cities.copy(), proj_params)
fmt = '{:,} / {:,} cities selected at random.'
print(fmt.format(cities_total, total_cities))

# routes, create
idx = pd.IndexSlice
distance = distance_matrix(x=cities[['x', 'y']], y=cities[['x', 'y']])
routes_histogram(distance, min_distance)
routes, routes_available = routes_define(cities, distance, min_distance, routes_total, endpoint_stdev)
orig, dest = routes['endpoint'] == 'orig', routes['endpoint'] == 'dest'
fmt = '{} cities make possible {:,} routes.'
print(fmt.format(cities_total, cities_total**2))
routes_removed = cities_total**2 - routes_available
pct_removed = round(routes_removed*100/cities_total**2)
fmt = '{:,} / {:,} ({}%) routes shorter than {:,} km, so they are removed.'
print(fmt.format(routes_removed, cities_total**2, pct_removed, min_distance))

# routes, cluster
start = time.time()
routes = routes_cluster(routes, orig, dest, cluster_params, color_rgb, reachabilities_per)
stop = time.time()
fmt = '{:,} routes drawn at random.'
print(fmt.format(routes_total))
fmt = '{:,} routes clustered ({} seconds).'
print(fmt.format(routes_total, round(stop-start, 2)))

# routes, reachability
start = time.time()
for group in routes['reachability_plot'].unique()[:reachability_plots]:
    mask = (routes['reachability_plot'] == group) & (routes['reachability'] != np.inf)
    columns = ['route', 'route_cluster', 'reachability', 'reachability_order',
               'reachability_color', 'reachability_plot']
    routes_reachability(routes.loc[mask, columns], group)
stop = time.time()
fmt = '{:,} reachability plots ({} seconds).'
print(fmt.format(reachability_plots, round(stop-start, 2)))

# clusters, geometric
start = time.time()
clusters = clusters_projection(routes, cluster_params, color_hex, proj)
clusters_geometric(clusters, hull_maps)
stop = time.time()
fmt = '{:,} hull maps ({} seconds).'
print(fmt.format(hull_maps, round(stop-start, 2)))

# routes, show
start = time.time()
routes_show(routes, routes_per)
stop = time.time()
fmt = '{:,} routes mapped ({} seconds).'
print(fmt.format(routes_per, round(stop-start, 2)))

# clusters, monochromatic
start = time.time()
clusters_total = clusters_monochromatic(clusters)
stop = time.time()
fmt = '{:,} monochromatic clusters mapped ({} seconds).'
print(fmt.format(clusters_total, round(stop-start, 2)))

# clusters, polychromatic
start = time.time()
clusters.reset_index(inplace=True)
clusters.set_index(['route_cluster', 'endpoint'], inplace=True)
clusters = clusters_partition(clusters, color_hex, hubs_per)
clusters_polychromatic(clusters)
stop = time.time()
fmt = '{:,} polychromatic clusters mapped ({} seconds).'
print(fmt.format(clusters_total, round(stop-start, 2)))

# hubs
page, clusters, routes = hub_page(clusters, routes)
page, rank, graph = hub_rank(page)
rank = rank_format(rank, page, clusters, color_hex)
edge = edge_define(graph)
edge = edge_triangle(edge)
edge = edge_format(edge)

# hubs, page
start = time.time()
hub_show(clusters, rank, edge)
stop = time.time()
fmt = 'PageRank map ({} seconds).'
print(fmt.format(round(stop-start, 2)))

# clusters, size
routes.index.get_level_values(level=0).value_counts().hist()
plt.title('Route Cluster Size')
plt.xlabel('Number of Routes')
plt.ylabel('Number of Clusters')
plt.savefig(directory + '/_cluster_sizes.png')
val = (routes_total - routes.index.get_level_values(level=0).isnull().sum()//2)
fmt = '{:,} / {:,} routes ({:,}%) selected by clustering.'
print(fmt.format(val, routes_total, np.round(val*100/routes_total, 2)))
fmt = '{:,} / {:,} routes ({:,}%) removed by clustering.'
print(fmt.format(routes_total - val, routes_total, np.round((routes_total - val)*100/routes_total, 2)))

