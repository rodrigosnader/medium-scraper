from geopy.distance import vincenty as geo_dist
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt
from matplotlib import style
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode
from urllib.request import urlopen
from pandas import DataFrame as dframe
from pandas import Series as series
import bs4 as bs
import http.cookiejar as cookiejar
import inspect
import json
import numpy as np
import pandas as pd
import pickle
import re
import requests
import time
import socket
import urllib
from scipy import stats
import folium

geolocator = Nominatim()


 
def chop(text, split1, split2):
    chopped = text.split(split1)[1].split(split2)[0]
    return chopped


def check_connection(hostname='www.google.com'):
    try:
        host = socket.gethostbyname(hostname)
        s = socket.create_connection((host, 80), 2)
        return True
    except:
        pass
        return False

def get_soup(source_address, timeout=10):
    hdr = {'User-Agent': 'Mozilla/5.0'}

    try:
        req = urllib.request.Request(source_address, headers=hdr)
    except:
        print('Could not complete urllib request.')
        pass
    try:
        source = urllib.request.urlopen(req, timeout=timeout).read()
        soup = bs.BeautifulSoup(source, 'lxml')
        return soup
    except:
        print('Could not create soup.')
        pass
    return


# Plot Functions


def show_on_map(s):
    '''
    Mostra o ponto de um dataframe no mapa
    
    Parameters
    ----------
    
    s - Series = df.iloc[x]
    
    '''
    
    s = s[['latitude', 'longitude', 'price_area']]
    hmap = folium.Map(location=[s.latitude, s.longitude], zoom_start=15,)

    lat = s.latitude
    lon = s.longitude
    price_area = s.price_area

    folium.Marker(
        location=[lat, lon],
        popup='R$' + str(price_area),
        icon=folium.Icon(color='blue', icon='circle'),
    ).add_to(hmap)

    return hmap


def dist_plot(df, figsize=[20,8]):
    for column in df.columns:
        try:
            if df[column].dtype == float:
                fig = plt.figure(figsize=figsize)
                plt.xlabel('index', fontsize=18)
                plt.ylabel(column, fontsize=18)
                plt.scatter(df.index, df[column])
                plt.show()
        except:
            pass
    return


def colorplot(x, y, s, c):
    fig = plt.figure(figsize=[20,8])
    x, y, s, c = x.values, y.values, s.values, c.values
    cm = plt.scatter(x=x, y=y, alpha=0.4, s=s, c=c, cmap=plt.get_cmap("jet"))
    plt.colorbar(cm)
    plt.show()


# Dataframe Functions

def to_array(df):

    array = df.values

    if len(array) == 1:
        array = array.reshape((array.shape[0], 1))

    return array


def generate_dummies(df):
    
    for column in df.columns:
        if df[column].dtypes == object:
            df = pd.concat([df, pd.get_dummies(df[column])], axis=1)
            df = df.drop(column, 1)
    return df


def percentile_filter(df, column, lim=1, verbose=1):
    
    min_lim = 0 + lim
    max_lim = 100 - lim
    old_len = len(df)
    low_limit = np.percentile(df[column].values, min_lim)
    high_limit = np.percentile(df[column].values, max_lim)

    if verbose == 1:
        print('----', column, '----')
        print('Min Limit:', low_limit, 'Max Limit:', high_limit)

    df = df[df[column] > low_limit]
    df = df[df[column] < high_limit]
    new_len = len(df)

    if verbose == 1:
        print('Dataframe Lenght:', old_len, '---->', new_len)
    
    return df


def decode_strings(df, mode=0):
    for column in df:
        if mode==0:
            if df[column].dtype == object:
                df[column] = df[column].str.decode('unicode_escape').str.encode('latin1').str.decode('utf8')
        if mode ==1:
            if df[column].dtype == object:
                df[column] = df[column].str.encode('latin1').str.decode('utf8')

    return df



# String Functions 

def str_to_numeric(s):
    try:
        s = re.split('\D', s)
        s = '.'.join(s)
        s = float(s)
    except:
        s=np.nan
    return s

def to_float(s, remove_dots=False, remove_commas=False):

    # Só funciona com números inteiros tipo mil: 1.000 ou um milhao 1.000.000

    try:
        s = re.findall(r'-?\+?\d+\.?,?\d*\.?,?\d*', str(s))[0]
        if remove_commas == True:
            if ',' in s:
                s = s.replace(',', '')
        if remove_dots == True:
            if '.' in s:
                s = s.replace('.', '')
        s = float(s)     
    except:
        s = None
    return s


def money_to_float(s):
    if type(s) == int or type(s) == float:
        s = str(s)
        
    try:
        s = re.findall(r'\d+\.?,?\d*\.?,?\d*', s)[0]
        if ',' in s:
            s = s.split(',')[0]
        if '.' in s:
            s = s.replace('.', '')
        s = float(s)
    except:
        s = None
    return s


def remove_range(s):
    try:
        s = re.findall(r'\d*-\d*', s)[0].split('-')[0]
    except:
        pass
    return s

def slash_split(s):
    try:
        s = s.split('/')[1].strip()
    except:
        return s
    return s

def number_split(s):
    try:
        s = re.split(r'\d*', s)[1].strip()
    except:
        return s
    return s

def area_split(s):
    try:
        s = re.split(r'Área', s)[1].strip()
    except:
        return s
    return s

def remove_space(s):
    s = s.replace(' ', '')
    return s


def str_norm(s):
    if pd.isnull(s) == False:

        normalized = text_norm(s)

        if ' ' in normalized:
            normalized = normalized.replace(' ', '-')

        try:
            if normalized[0] == '-':
                normalized = normalized[1:]
        except:
            pass
        try:
            if normalized[-1] == '-':
                normalized = normalized[:-1]
        except:
            pass

        return normalized
    
def text_norm(s):
    if pd.isnull(s) == False:
        normalized = unidecode(s.lower()).strip()
        if "'" in normalized:
            normalized = normalized.replace("'", "")
            normalized = normalized.strip()
        if "-" in normalized:
            normalized = normalized.replace("-", " ")
            normalized = normalized.strip()
        if "\n" in normalized:
            normalized = normalized.replace("\n", "")
            normalized = normalized.strip()
        if "\r" in normalized:
            normalized = normalized.replace("\r", "")
            normalized = normalized.strip()

        normalized = re.sub(' +',' ',normalized)

        return normalized



# Cluster Functions

def get_outliers(s, eps=0.8, min_samples=5):

    '''
    DBSCAN para identificar, vizualizar e remover outliers

    '''

    try:
        dim = len(s.columns)
    except:
        dim = 1
    s = s.dropna()
    x = s.values.reshape(len(s), dim)
    x = StandardScaler().fit_transform(x)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    model = dbscan.fit(x)
    return series(model.labels_ != stats.mode(model.labels_).mode[0], index=s.index)


def plot_outliers(s, eps=0.8, min_samples=5):

    style.use('seaborn-deep')

    # Scatter as duas primeiras colunas caso s seja um dataframe
    try:
        dim = len(s.columns)
    except:
        dim = 1

    outliers = get_outliers(s, eps=eps, min_samples=min_samples)
    fig = plt.figure(figsize=[20,8])
    
    if dim == 1:
        s = s[outliers.index]
        plt.scatter(s.index, s, c=outliers.values, cmap=plt.get_cmap("bwr"), alpha=0.5)
    else:
        s = s.loc[outliers.index]
        plt.scatter(s[s.columns[0]], s[s.columns[1]], c=outliers.values, cmap=plt.get_cmap("bwr"), alpha=0.5)
    plt.show()
    

def remove_outliers(s, eps=0.8, min_samples=5, limit=0.05):
    try:
        dim = len(s.columns)
    except:
        dim = 1
    outliers = get_outliers(s, eps=eps, min_samples=min_samples)
    if len(outliers[outliers == True]) > len(s)*limit:
        return s
    else:
        if dim == 1:
            s = s[outliers.index]
        else:
            s = s.loc[outliers.index]
        s = s[outliers == False]

    return s


def show_outliers(s, link=None, eps=0.8, min_samples=5):

    plot_outliers(s, eps=eps, min_samples=min_samples)
    outliers = get_outliers(s, eps=eps, min_samples=min_samples)[get_outliers(s, eps=eps, min_samples=min_samples) == True]

    indexes = None

    try:
        indexes = link[outliers.index]
    except:
        pass

    return indexes



# Model Functions

def prepare(data, target_col='price_area', normalization=None):

  features = data.drop(target_col, 1)
  target = data[target_col]
  x = to_array(features)

  y = to_array(target)

  if normalization == 'normal':
    
      normalizer = Normalizer()
      x = normalizer.fit_transform(x)
    
  if normalization == 'standard':
    
      standardizer = StandardScaler()
      x = standardizer.fit_transform(x)

  if normalization == 'robust':
    
      robuster = RobustScaler()
      x = robuster.fit_transform(x)

  return x, y

def train_test_split(data, test_size=0.2):

    split_size = int(math.floor(len(data) * test_size))

    train = data[split_size:]
    test = data[:split_size]

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    return train, test


def compare(estimator, x_test, y_test, n_disp=5):

    original = y_test
    prediction = estimator.predict(x_test).flatten()
    comp = {'original': original, 'predicted': prediction, 'error': original - prediction}
    comp_df = pd.DataFrame(comp)
    comp_df['percent_err'] = np.abs((comp_df.original - comp_df.predicted)/comp_df.original)
    mape = np.mean(comp_df.percent_err)
    medape = np.median(comp_df.percent_err)
    std = np.std(comp_df.percent_err)
    samples = comp_df[:n_disp]
    stats = {'Medape': medape, 'Mape': mape, 'Std': std, 'Highest Error': np.max(comp_df.percent_err.values)}
    best = comp_df.iloc[comp_df.percent_err.nsmallest(n_disp).index.tolist()]
    worst = comp_df.iloc[comp_df.percent_err.nlargest(n_disp).index.tolist()]
    desc = comp_df.percent_err.describe()
    
    return stats, samples, best, worst, desc




# Helper Functions

def reencode(path):

    with open(path, 'rb') as file:
        reencoded = file.read().decode('UTF-8', 'replace').encode()
        with open(path, 'wb') as file_out:
            file_out.write(reencoded)

    return


def var_name(var):
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]


def read_pickle(file):
    with open(file, 'rb') as handle:
         load = pickle.load(handle)
    return load

def to_pickle(var, dir):
    directory = dir + '.pickle'
    with open(directory, 'wb') as handle:
        pickle.dump(var, handle)
    return

def reverse_distance_mean(values, distances, w=1):
    values = np.array(values)
    distances = np.array(distances) 
    distances = np.power(distances, w)
    ones = np.ones(shape=len(values))
    
    sup = values/distances
    inf = ones/distances
    
    return sup.sum()/inf.sum()









