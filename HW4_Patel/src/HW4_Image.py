import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from timeit import default_timer as tm
from sklearn.manifold import Isomap,TSNE

start = tm()
clusters = 10
image = pd.read_csv('Image_Test.txt', header = None)

sse={}
for k in range(2, 20, 2):
    kmeans = KMeans(n_clusters=k, max_iter=150).fit(image)
    image["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

image = MinMaxScaler().fit_transform(image)

image = Isomap(n_components=70).fit_transform(image)

image_np = np.array(image)
image_list = image.tolist()

# for row in range(len(image_list)):
#     for col in range(len(image_list[row])):
#         image_list[row][col] = image_list[row][col]/255.0
# print("Normalized!")

idx = np.random.choice(len(image_list), clusters, replace=False)
centroids = image_np[idx, :]
# centroids = image_np[idx]

distances = cdist(image, centroids, 'cosine')

points = np.array([np.argmin(i) for i in distances])

for _ in range(len(image)):
    centroids = []
    for idx in range(clusters):
        temp_cent = image[points == idx].mean(axis=0)
        centroids.append(temp_cent)

    centroids = np.vstack(centroids)
    distances = cdist(image, centroids, 'cosine')
    points = np.array([np.argmin(i) for i in distances])

print(points)
points = points.tolist()

pca = PCA(2)

df = pca.fit_transform(image)

df_center = pca.fit_transform(centroids)

plt.scatter(df[:, 0], df[:, 1], c=points,s=40, cmap='viridis')
plt.scatter(df_center[:, 0], df_center[:, 1], c='black', s=200, alpha=0.5)
plt.show()
#writing in the output file
output_file = open("image_output2.txt", "w+")
for i in points:
    output_file.write(str(i+1)+"\n")

print(tm() - start)