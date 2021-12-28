import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
clusters = 3

iris = pd.read_csv('Iris_Test_data.txt', sep=" ", names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])

iris.plot(kind="scatter", x="SepalLength", y="SepalWidth")
plt.show()

data = iris[['SepalLength', 'SepalWidth', 'PetalLength']]
sse={}
for k in range(2, 20, 2):
    kmeans = KMeans(n_clusters=k, max_iter=150).fit(data)
    data["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


iris_np = np.array(iris)
iris_list = iris.values.tolist()
print(len(iris_list))

idx = np.random.choice(len(iris_list), clusters, replace=False)
centroids = iris_np[idx, :]
print(centroids)

distances = cdist(iris, centroids, 'cosine')

points = np.array([np.argmin(i) for i in distances])

for _ in range(len(iris)):
    centroids = []
    for idx in range(clusters):
        temp_cent = iris[points == idx].mean(axis=0)
        centroids.append(temp_cent)

    centroids = np.vstack(centroids)
    distances = cdist(iris, centroids, 'cosine')
    points = np.array([np.argmin(i) for i in distances])

print(points)
points = points.tolist()

pca = PCA(2)

df = pca.fit_transform(iris)

df_center = pca.fit_transform(centroids)

plt.scatter(df[:, 0], df[:, 1], c=points,s=40, cmap='viridis')
plt.scatter(df_center[:, 0], df_center[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# writing in the output file
# output_file = open("iris_output.txt", "w+")
# for i in points:
#     output_file.write(str(i+1)+"\n")