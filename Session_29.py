#importing libraries
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Visualize raw image
f = misc.face(gray=True)
plt.figure(figsize=(10, 3.6))
plt.subplot(131)
plt.imshow(f, cmap=plt.cm.gray)
plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05,
                    right=0.99)
plt.show()

#Compressing the gray scale image into 5 clusters
rows = f.shape[0]
cols = f.shape[1]

#print(rows,cols)
image = f.reshape(rows*cols,1)
kmeans = KMeans(n_clusters = 5)
kmeans.fit(image)
clusters = np.asarray(kmeans.cluster_centers_) 
labels = np.asarray(kmeans.labels_)  
labels = labels.reshape(rows,cols); 

#np.save('codebook_racoon.npy',clusters)
plt.imsave('compressed_racoon.png',labels);

#Visualize the compressed image
image = plt.imread('compressed_racoon.png')
plt.figure(figsize=(10, 3.6))
plt.imshow(image)
plt.show()
