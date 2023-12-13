import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import datetime
import sys
import random
import math
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, silhouette_score
import seaborn as sns
import folium
from folium.plugins import HeatMap





# map qualitative descriptions of features to numerical values
def encodeEntries(df):
    #1.1 map categorical variables to numbers
    ic_mapping = {'Wifi  / local Internet service provider': 3,
                'Mobile network - 3G / 4G': 2,
                'No Internet access': 0,
                'Unknown': 1}  # Replace 'nan' with an appropriate numerical value
    df['Type of Internet Connection'] = df['Type of Internet Connection'].map(ic_mapping)
    # print(df['Type of Internet Connection'].value_counts() )

    # 1.2 
    water_waste_mapping = {
        'Direct discharge to environment': 0,
        'Unknown': 1,
        'Cesspit': 2,
        'Open pit': 3,
        'Holding tank': 4,
        'Municipality sewer network / not treated': 5,
        "Storm water channel": 6,
        "Septic tank": 7,
        "Municipality sewer network / treated": 8,
        "Irrigation canal": 9 }
    df['Waste Water Disposal'] = df['Waste Water Disposal'].map(water_waste_mapping)

    # 1.3
    vacc_mapping = {'Yes':2, 
                    'Unknown':1,
                    'No':0}
    df['Free Vaccination for Children under 12'] = df['Free Vaccination for Children under 12'].map(vacc_mapping)
    # 1.4
    waste_disposal_mapping = {
        "Municipality Collection": 4,  # Assumes proper waste management
        'Unknown': 2,  # Uncertain impact
        "Burn it": 1,  # Air pollution and emissions
        "Dump it outside the camp": 0,  # Negative impact on local environment
        "Burry it": 3  # Potential soil contamination
    }
    df['Waste Disposal'] = df['Waste Disposal'].map(waste_disposal_mapping)
    #1.5
    water_source_mapping = {
        "Unknown": 0,
        "Water Trucking": 2,
        "Borehole": 6,
        "Well": 5,
        "Spring": 4,
        "River": 3,
        "Water Network": 7,
        "Others": 1,
    }
    df['Type of Water Source'] = df['Type of Water Source'].map(water_source_mapping)
    #1.6
    status_mapping = {'Active': 2,
                    'Less than 4':1,
                    'Inactive':0,
                    'Unknown':0,
                    'Erroneous':0,
                    'Not Willing':0}
    df['Status'] = df['Status'].map(status_mapping)

    return df

df = pd.read_csv('informalsettlements.csv')
df = df.fillna('Unknown')
df = df.drop(['Pcode\n','Governorate', 'District', 'Cadaster', 'Type of Contract','Local Name','Shelter Type','Date of the Update', 'Updated By', 'Updated On','Discovery Date', 'Date the site was created','Consultation Fee for PHC (3000/5000)'],axis='columns')

label_encoder = LabelEncoder()
# df['Pcode Name'] = label_encoder.fit_transform(df['Pcode Name'])
headers =df.keys()
print(headers)
df = encodeEntries(df)
# Split into training (70%) and temporary set (30%)
train_data, validation_data = train_test_split(df, test_size=0.85, random_state=42)
# # Save training, validation, test data
train_data.to_csv('p2_train.csv', index=False, header=headers)
validation_data.to_csv('p2_validation.csv', index=False, header=headers)

# calculates the correlation between the top 10-rated settlements in gaussian and k-means
# returns a 10 element vector of Pearson coefficients
def calculate_correlations(a, b):
    # Ensure that the arrays have the same shape
    correlations = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        # Calculate Pearson correlation coefficient for each pair of vectors
        correlations[i], _ = pearsonr(a[i], b[i])
    return correlations

# GMM model
sc = MinMaxScaler()
X_train=sc.fit_transform(train_data.values[:,3:])
X_val =sc.fit_transform(train_data.values[:,3:])
n_components_g = 65
n_components_k = 64

# determine similarities and choose n_components based off the highest pearson correlation
# coefficient between sorted cluster means between the GMM and the K-means
# use Pearson and cosine similarities
def hyperparametertuning(X_train, X_val): 
    best_coeff = -1
    best_n_components_k = -1
    best_n_components_g = -1
    np.set_printoptions(suppress=True)
    # find optimal number of clusters k in the data and optimal number of gaussians 
    for n_components_g in range(10,120,1):
        gmm = GaussianMixture(n_components=n_components_g,covariance_type='tied', tol=0.0001,init_params='kmeans', n_init=100)
        gmm.fit(X_train)
        cluster_means_gmm = gmm.means_
        gmm_norms = np.linalg.norm(cluster_means_gmm, axis=1)
        sorted_indices = np.argsort(gmm_norms)[::-1]  # Sort in descending order
        sorted_cluster_means_gmm = cluster_means_gmm[sorted_indices]
        # print(f"GMM: {np.round(sorted_cluster_means_gmm[:3],2)}")
        for n_components_k in range(10,100,1):
            kmeans = KMeans(n_clusters=n_components_k, n_init = "auto", random_state=50)
            kmeans.fit(X_train)
            cluster_means_kmeans = kmeans.cluster_centers_
            norms_kmeans = np.linalg.norm(cluster_means_kmeans, axis=1)
            sorted_indices_kmeans = np.argsort(norms_kmeans)[::-1]  # Sort in descending order
            sorted_cluster_means_kmeans = cluster_means_kmeans[sorted_indices_kmeans]
            # print(f"kmeans: {np.round(sorted_cluster_means_kmeans[:3],2)}")
            curr_coeff =calculate_correlations(sorted_cluster_means_gmm[:10],sorted_cluster_means_kmeans[:10])
            # take the mean of the correlation coefficients for the top 10 clusters/gaussians and compare them
            # print(f"n_comp_k: {n_components_k}, curr_coeff: {curr_coeff}")
            if (np.mean(curr_coeff)> best_coeff): 
                best_coeff = np.mean(curr_coeff)
                best_n_components_k = n_components_k
                best_n_components_g = n_components_g
        # print(best_coeff)
    print(f"Optimized n_components_k: {best_n_components_k}")
    print(f"Optimized n_components_g: {best_n_components_g}")
    print(f"Max Pearson correlation coefficient: {best_coeff}")

# hyperparametertuning(X_train, X_val)

n_components_g = 64
n_components_k = 65

gmm = GaussianMixture(n_components=n_components_g,covariance_type='tied', tol=0.0001,init_params='kmeans', n_init=100)
gmm.fit(X_train)
val_predictions = gmm.predict(X_val)
silhouette_val = silhouette_score(X_val, val_predictions)
print(f"GMM silhoutte score:{silhouette_val}")
cluster_means_gmm = gmm.means_
gmm_norms = np.linalg.norm(cluster_means_gmm, axis=1)
sorted_indices = np.argsort(gmm_norms)[::-1]  # Sort in descending order
sorted_norms = gmm_norms[sorted_indices]
sorted_cluster_means = sc.inverse_transform(cluster_means_gmm[sorted_indices])

# # Print or use the means
np.set_printoptions(suppress=True)
print("GMM Cluster Means:")
print(np.round(sorted_norms[:5], 2))
print(np.round(sorted_cluster_means[:5],2))
print("Bottom five clusters:")
print(np.round(sorted_norms[-5:], 2))

# # K-means clustering
kmeans = KMeans(n_clusters=n_components_k, n_init = "auto", random_state=50)
kmeans.fit(X_train)
val_predictions_kmeans = kmeans.predict(X_val)
silhouette_val_kmeans = silhouette_score(X_val, val_predictions_kmeans)
print("Silhouette Score (KMeans):", silhouette_val_kmeans)

cluster_means_kmeans = kmeans.cluster_centers_
norms_kmeans = np.linalg.norm(cluster_means_kmeans, axis=1)
sorted_indices_kmeans = np.argsort(norms_kmeans)[::-1]  # Sort in descending order
sorted_norms_kmeans = norms_kmeans[sorted_indices_kmeans]
sorted_cluster_means_kmeans = sc.inverse_transform(cluster_means_kmeans[sorted_indices_kmeans])

train_clusters = gmm.predict(X_train)
#Print or use the means
print("Cluster Means (KMeans):")
np.set_printoptions(suppress=True)
print(np.round(sorted_norms_kmeans[:5], 2))
print(np.round(sorted_cluster_means_kmeans[:5], 2))
print("Bottom five clusters (KMeans):")
print(np.round(sorted_norms_kmeans[-5:], 2))

train_data['Cluster'] = train_clusters
cluster_scores_mapping = dict(zip(sorted_indices, sorted_norms))
train_data['Score'] = train_data['Cluster'].map(cluster_scores_mapping)
print(train_data[['Pcode Name', 'Score']])


# Create a Folium map centered around the average latitude and longitude of your data
# average_lat = train_data['Latitude'].mean()
# average_lon = train_data['Longitude'].mean()
# print(f"mean: {train_data['Score'].mean()}")
# print(f"25 quantile: {train_data['Score'].quantile(0.25)}")
# print(f"75 quantile: {train_data['Score'].quantile(0.75)}")

# map_center = [train_data['Latitude'].mean(), train_data['Longitude'].mean()]
# intensity_map = folium.Map(location=map_center, zoom_start=12)




# Assuming 'Latitude', 'Longitude', and 'Score' are columns in your train_data DataFrame
# Replace 'Score' with the actual column name representing the intensity

from branca.colormap import LinearColormap
# Assuming you already have train_data with 'Latitude', 'Longitude', 'Score', and 'Pcode Name' columns

# Create a colormap for the scores
colormap = LinearColormap(['red', 'yellow', 'green'], vmin=train_data['Score'].min(), vmax=train_data['Score'].max())

# Create a folium map with satellite imagery
my_map = folium.Map(location=[train_data['Latitude'].mean(), train_data['Longitude'].mean()], zoom_start=9, control_scale=True)
# folium.TileLayer('MapQuest Open Aerial').add_to(my_map)
# folium.LayerControl().add_to(my_map)
# Add markers for each data point
for index, row in train_data.iterrows():
    lat, lon, score, pcode_name = row['Latitude'], row['Longitude'], row['Score'], row['Pcode Name']
    # Get the color based on the score
    color = colormap(score)
     # Adjust radius and fill opacity based on the score
    radius = 8 * score  # You can adjust the multiplier to control the size
    fill_opacity = 0.5 + 0.2 * score  # You can adjust the multiplier to control the opacity
    # Create a CircleMarker with the corresponding color, radius, and fill opacity
    folium.CircleMarker(location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color,
                        fill_opacity=fill_opacity, popup=f'{pcode_name}<br>Score: {score}', icon=folium.Icon(color=color, icon='info')).add_to(my_map)
    # # Create a CircleMarker with the corresponding color
    # folium.CircleMarker(location=[lat, lon], radius=8, color=color, fill=True, fill_color=color,
    # fill_opacity=0.7,popup=f'{pcode_name}<br>Score: {score}').add_to(my_map)

# Add a legend
colormap.caption = 'Score Legend'
colormap.add_to(my_map)

# Add layer control to toggle the legend
folium.LayerControl().add_to(my_map)


# Display the map
my_map
my_map.save('intensity_map.html')


# label each datapoint with its cluster label from GMM, then use the latitude, longitude, cluster numbers, 
# and form a color map based on how high the numbers are 
#visualize this on openstreetmaps

# clusters = gmm.predict(data)
# # Add the cluster assignments to your DataFrame
# df['Cluster'] = clusters

# # Print the cluster assignments
# print(df[['Pcode Name', 'Cluster']])