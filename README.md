# Geospatial Clustering for Delivery Optimization

This project explores the application of geospatial clustering techniques to optimize delivery logistics using a real-world dataset. The goal is to group delivery pickup and drop-off locations into meaningful clusters based on their geographical proximity, enabling smarter location-based decisions for logistics, resource allocation, and urban planning.

## Table of Contents
1.  [Problem Statement](#Problem-Statement)
2.  [Dataset Overview](#Dataset-Overview)
3.  [Distance Calculation](#Distance-Calculation)
4.  [Interactive Visualization of Delivery Locations](#Interactive-Visualization-of-Delivery-Locations)
5.  [K-Means Clustering](#K-Means-Clustering)
6.  [Outlier Removal and Zone Optimization](#Outlier-Removal-and-Zone-Optimization)
7.  [Conclusion](#Conclusion)

## 1. Problem Statement
Geospatial clustering is a powerful unsupervised learning technique used to identify patterns and groupings within spatial data. In the context of delivery services, this can involve creating efficient delivery zones, identifying high-demand areas, or optimizing routes. This project aims to demonstrate how K-Means clustering can be applied to delivery location data to achieve such optimizations, specifically by identifying distinct delivery zones and handling geographical outliers.

## 2. Dataset Overview
The dataset used for this project is sourced from Kaggle and contains detailed information about various delivery orders, including geographical coordinates for restaurants and delivery locations. Initial data loading and exploration are performed using the `pandas` library.


## import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic

data = pd.read_csv("/content/train.csv")
print(data.head())


## 3. Distance Calculation
To understand the actual travel required for deliveries, the real-world distance (in kilometers) between each restaurant and its corresponding delivery location is calculated using the `geodesic` formula from the `geopy` library. This provides a more accurate measure than Euclidean distance, accounting for the curvature of the Earth.

...def calculate_distance(row):
    return geodesic(
        (row['Restaurant_latitude'], row['Restaurant_longitude']),
        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    ).km

data['Distance_km'] = data.apply(calculate_distance, axis=1)
print(data[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Distance_km']].head())

## 4. Interactive Visualization of Delivery Locations
All delivery locations across India are visualized on an interactive map using Plotly. This initial visualization helps to understand the geographical distribution of delivery activities and identify areas of high concentration or sparsity.


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=data['Delivery_location_longitude'],
    lat=data['Delivery_location_latitude'],
    mode='markers',
    marker=dict(color='blue', size=6, opacity=0.7),
    name='Delivery Locations',
    hovertemplate='Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra>Delivery</extra>'
))

fig.update_layout(
    title='📦 Mapping Our Reach — Delivery Locations Across India',
    geo=dict(
        scope='asia',
        showland=True,
        landcolor='rgb(229, 229, 229)',
        showcountries=True,
        countrycolor='rgb(200, 200, 200)',
        showlakes=False,
        lonaxis=dict(range=[68, 98]),  # focus on India
        lataxis=dict(range=[6, 38])
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    showlegend=False
)

fig.show()

The visualization reveals that delivery activity is predominantly concentrated in the southern and central regions of India, with fewer points in the northern and northeastern zones. This insight can inform strategic service expansion or resource allocation decisions.

## 5. K-Means Clustering
K-Means clustering is applied to the delivery location coordinates to group them into a predefined number of clusters (`k=3` in this case). The clusters and their centroids are then visualized on a map, allowing for an immediate understanding of the identified geographical groupings.

X = data[['Delivery_location_latitude', 'Delivery_location_longitude']]
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init to suppress warning
data['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

fig = go.Figure()

for cluster_label in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_label]
    fig.add_trace(go.Scattergeo(
        lon=cluster_data['Delivery_location_longitude'],
        lat=cluster_data['Delivery_location_latitude'],
        mode='markers',
        name=f'Cluster {cluster_label}',
        marker=dict(size=6, opacity=0.7),
        hovertemplate='<b>Cluster:</b> %{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
        text=[f"{cluster_label}"] * len(cluster_data)
    ))

fig.add_trace(go.Scattergeo(
    lon=centroids[:, 1],
    lat=centroids[:, 0],
    mode='markers',
    name='Centroids',
    marker=dict(size=15, symbol='x', color='red', line=dict(width=2, color='black')),
    hovertemplate='<b>Centroid</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
))

fig.update_layout(
    title=f'📍 Geo-Spatial Clustering of Delivery Locations (k = {k})',
    geo=dict(
        scope='asia',
        showland=True,
        landcolor="rgb(229, 229, 229)",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)",
        lonaxis=dict(range=[68, 98]),
        lataxis=dict(range=[6, 38]),
    ),
    legend_title='Clusters',
    margin=dict(l=0, r=0, t=60, b=0)
)

fig.show()

## 6. Outlier Removal and Zone Optimization
Upon inspecting the clustered results, it was observed that one of the clusters (Cluster 1) contained geographical points outside the expected Indian boundaries, indicating potential data entry errors or GPS inaccuracies. This outlier cluster is removed, and the remaining valid clusters are relabeled with business-contextual names like "Central Delivery Zone" and "Southern Delivery Zone" to make them more actionable for logistics planning.


filtered_data = data[data['Cluster'] != 1]
filtered_centroids = centroids[[0, 2]]  # Keep only Cluster 0 and 2

cluster_labels = {
    0: "Central Delivery Zone",
    2: "Southern Delivery Zone"
}
filtered_data['Optimized_Zone'] = filtered_data['Cluster'].map(cluster_labels)

print(filtered_data['Optimized_Zone'].value_counts())

## 7. Conclusion
This project successfully demonstrates how geospatial clustering can be used to segment delivery locations into meaningful zones. By identifying and removing outliers, and then assigning intuitive names to the clusters, the raw spatial data is transformed into actionable intelligence for improved logistics planning, such as optimizing delivery routes, allocating resources efficiently, and identifying strategic areas for business expansion. This approach provides a solid foundation for more advanced location-based decision-making.
