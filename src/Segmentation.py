import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from tqdm import tqdm
from heapq import heappush, heappop
import cv2
from collections import deque

class Agglomerative_Clustering:
    def __init__(self, linkage_type='complete'):
        self.linkage_type = linkage_type
        
    def min_value_not_on_diagonal(self, matrix):
        min_value = float('inf')
        min_x, min_y = -1, -1
        
        # Iterate through the elements of the matrix
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                # Check if the current element is not on the main diagonal
                if i != j:
                    # Update the minimum value and its indices if needed
                    if matrix[i][j] < min_value:
                        min_value = matrix[i][j]
                        min_x = i
                        min_y = j
        
        return min_value, min_x, min_y
        

    def cluster_distance(self, X, cluster_members):
        """
        Calculate the distance between clusters.

        Parameters
        ----------
        X: Dataset, shape (nSamples, nFeatures)
        cluster_members: Dictionary mapping cluster keys to their member indices

        Returns
        -------
        2D array of distances between clusters
        """
        nClusters = len(cluster_members)
        keys = list(cluster_members.keys())
        centroids = np.array([X[cluster_members[key]].mean(axis=0) for key in keys])
        Distance = np.zeros((nClusters, nClusters))
        
        for i in range(nClusters):
            for j in range(i+1, nClusters):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                Distance[i, j] = dist
                Distance[j, i] = dist
        return Distance
    
    def cluster_distance(self, X, cluster_members):
        """
        Calculates the cluster distances based on the specified linkage type.

        Params
        ------
        cluster_members: dict
            Stores the cluster members in format: {key: [item1, item2 ..]}.
            If key is less than X.shape[0], then it only has itself in the cluster.

        Returns
        -------
        Distance: 2D array
            Contains distances between each cluster.
        """
        n_clusters = len(cluster_members)
        keys = list(cluster_members.keys())
        Distance = np.zeros((n_clusters, n_clusters))

        # Compute pairwise distances between clusters
        for i in range(n_clusters):
            ith_elems = cluster_members[keys[i]]
            for j in range(i + 1, n_clusters):
                jth_elems = cluster_members[keys[j]]
                d_in_clusters = euclidean_distances(X[ith_elems], X[jth_elems])
                if self.linkage_type == 'complete':
                    Distance[i, j] = np.max(d_in_clusters)
                elif self.linkage_type == 'single':
                    Distance[i, j] = np.min(d_in_clusters)
                # Since the distance matrix is symmetric, we can assign the same value to the opposite side
                Distance[j, i] = Distance[i, j]

        return Distance


    def fit(self, X):
        """
        Generates the dendrogram.

        Params
        ------
        X: Dataset, shape (nSamples, nFeatures)

        Returns
        -------
        Z: 2D array. shape (nSamples-1, 4). 
            Linkage matrix. Stores the merge information at each iteration.
        """
        self.nSamples = X.shape[0]
        cluster_keys = list(range(self.nSamples))
        cluster_members = {i: [i] for i in cluster_keys}
        Z = np.zeros((self.nSamples-1,4)) # c1, c2, d, count

        with tqdm(total=self.nSamples-1) as pbar:
            for i in range(0, self.nSamples-1):
                pbar.update(1)  # Increment progress bar by one step

                keys = list(cluster_members.keys())
                # caculate the distance between existing clusters
                D = self.cluster_distance(X,cluster_members)
                
                # Using heap to find minimum value
                min_heap = []
                for x in range(len(D)):
                    for y in range(x+1, len(D)):
                        heappush(min_heap, (D[x, y], x, y))

                _, tmpx, tmpy = heappop(min_heap)

                x = keys[tmpx]
                y = keys[tmpy]
                # update Z
                Z[i,0] = x
                Z[i,1] = y
                Z[i,2] = D[tmpx, tmpy] # that's where the min value is
                Z[i,3] = len(cluster_members[x]) + len(cluster_members[y])

                # new cluster created
                cluster_members[i+self.nSamples] = cluster_members[x] + cluster_members[y]
                # remove merged from clusters pool, else they'll be recalculated
                del cluster_members[x]
                del cluster_members[y]

        self.Z = Z
        return self.Z

    
    def predict(self, num_cluster=3):
        """
        Get cluster label for specific cluster size.
        
        Params
        ------
        num_cluster: int. 
            Number of clusters to keep. Can not be > nSamples
        
        Returns
        -------
        labels: list.
            Cluster labels for each sample.
        """
        labels = np.zeros(self.nSamples, dtype=int)  # Initialize labels array
        cluster_members = {i: [i] for i in range(self.nSamples)}  # Initialize clusters
        
        # Iterate until desired number of clusters is reached
        for i in range(self.nSamples - num_cluster):
            x, y = int(self.Z[i, 0]), int(self.Z[i, 1])  # Get clusters to merge
            cluster_members[self.nSamples + i] = cluster_members[x] + cluster_members[y]  # Merge clusters
            del cluster_members[x]  # Remove merged clusters
            del cluster_members[y]
        
        # Assign labels to samples based on the final clusters
        for label, samples in enumerate(cluster_members.values()):
            labels[samples] = label
            
        return labels


class KMeans:
    """
    KMeans clustering algorithm for clustering n-dimensional data.

    Attributes:
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations for the algorithm.
        centroids (list): List of centroids, initialized after first iteration.
    """
    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        self.centroids = []

    def predict(self, X):
        """
        Performs K-means clustering on the data X.

        Args:
            X (np.ndarray): The input data array of shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of cluster labels.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Initialize centroids
        random_indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            clusters = self._create_clusters(self.centroids)
            new_centroids = self._calculate_centroids(clusters)
            
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        return self._get_cluster_labels(clusters)

    def _create_clusters(self, centroids):
        distances = np.sqrt(((self.X - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_centroids = np.argmin(distances, axis=0)
        return {i: np.where(closest_centroids == i)[0] for i in range(self.K)}

    def _calculate_centroids(self, clusters):
        return np.array([self.X[cluster].mean(axis=0) for cluster in clusters.values()])

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, samples in clusters.items():
            labels[samples] = cluster_idx
        return labels
    

class MeanShiftSegmentation:
    def __init__(self, image):
        self.image = image

    def segment_image(self):
        """
        Segments the image using the mean shift algorithm.

        Returns:
            segmented_image (ndarray): The segmented image with the same shape as the input image,
                where each pixel is assigned a color representing the cluster it belongs to.
        """

        image_height, image_width, _ = self.image.shape
        segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        color_list = self.image.reshape(-1, 3)
        position_list = np.indices((image_height, image_width)).reshape(2, -1).T
        color_and_position_list = np.hstack((color_list, position_list))

        # Random mean
        current_mean = np.zeros(5)
        current_mean = color_and_position_list[np.random.randint(0, color_and_position_list.shape[0])]

        while color_and_position_list.shape[0] > 0:
            # Calculate distances between each pixel color and the current mean
            distances = self.get_distances(color_and_position_list[:, :3], current_mean)
            distances = np.where(distances < 100)[0]  # Indices

            # Calculate new mean
            mean_color = np.mean(color_and_position_list[distances, :3], axis=0)
            mean_position = np.mean(color_and_position_list[distances, 3:], axis=0)

            # Calculate color and position distances to new mean
            color_distance_to_mean = np.sqrt(np.sum((mean_color - current_mean[:3]) ** 2))
            position_distance_to_mean = np.sqrt(np.sum((mean_position - current_mean[3:]) ** 2))

            # Calculate total distance
            total_distance = color_distance_to_mean + position_distance_to_mean

            if total_distance < 200:  # Threshold
                # Update color of pixels in cluster
                new_color = np.zeros(3)
                new_color = mean_color
                segmented_image[
                    color_and_position_list[distances, 3],
                    color_and_position_list[distances, 4],
                ] = new_color
                # Remove pixels from list
                color_and_position_list = np.delete(color_and_position_list, distances, axis=0)

                # New random mean
                if color_and_position_list.shape[0] > 0:
                    current_mean = color_and_position_list[np.random.randint(0, color_and_position_list.shape[0])]

            else:
                # Update current mean
                current_mean[:3] = mean_color
                current_mean[3:] = mean_position

        return segmented_image


    def get_distances(self, color_and_position_list, current_mean):
        """
        Calculate the distances between each pixel color and the current mean

        Parameters
        ----------
        color_and_position_list : ndarray
            A list of pixel colors and their corresponding positions
        current_mean : ndarray
            The current mean color

        Returns
        -------
        ndarray
            The distances between each pixel color and the current mean
        """
        distances = np.zeros(color_and_position_list.shape[0])
        for i in range(len(color_and_position_list) - 1):
            # Calculate the distance between each pixel color and the current mean
            distance = 0
            for j in range(3):
                distance += (current_mean[j] - color_and_position_list[i][j]) ** 2
            # Square root of the distance
            distances[i] = distance ** 0.5
        return distances
    
class RegionGrowing:
    def __init__(self, image, seeds):
        self.image = image
        self.seeds = seeds

    def region_growing(self, threshold = 100, window_size = 20):
        """
        Region growing algorithm.
        Algorithm:
            1. Convert input image to grayscale
            2. Create a boolean mask to keep track of visited pixels
            3. Create a copy of the input image to store the segmented image
            4. For each seed point:
                1. Compute the mean color of the region
                2. Create a queue to store neighboring pixels
                3. Add the seed point to the queue
                4. While the queue is not empty:
                    1. Pop the current pixel from the queue
                    2. If the current pixel is not visited and its color is similar to the mean color:
                        1. Update the mean color
                        2. Mark the pixel as visited and part of the region
                        3. Add its neighboring pixels to the queue
                5. Return the segmented image
        :param threshold: The threshold for color similarity
        :param window_size: The size of the neighborhood to consider
        :return: The segmented image
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Create a boolean mask to keep track of visited pixels
        visited = np.zeros_like(gray_image, dtype=bool)

        # Create a copy of the input image to store the segmented image
        segmented = np.zeros_like(self.image)

        # Half window size
        half_window = window_size // 2

        for seed in self.seeds:
            x_seed, y_seed = seed[0], seed[1]
            if 0 <= x_seed < gray_image.shape[0] and 0 <= y_seed < gray_image.shape[1]:
                # Compute the mean color of the region
                mean_value = gray_image[x_seed, y_seed]

                # Number of pixels in the region (including the seed)
                number_of_pixels_in_region = 1

                # Create a queue to store neighboring pixels
                queue = deque([(x_seed, y_seed)])

                while len(queue):
                    current_x, current_y = queue.popleft()
                    if (0 <= current_x < gray_image.shape[0] and 
                        0 <= current_y < gray_image.shape[1]) and \
                        not visited[current_x, current_y]:
                        # Mark the pixel as visited
                        visited[current_x, current_y] = True

                        # Check if the color of the current pixel is similar to the mean color
                        if abs(gray_image[current_x, current_y] - mean_value) < threshold:
                            # Update the mean color
                            mean_value = ((mean_value * number_of_pixels_in_region + gray_image[current_x, current_y]) /
                                          (number_of_pixels_in_region + 1))
                            number_of_pixels_in_region += 1

                            # Mark the pixel as part of the region
                            segmented[current_x, current_y] = self.image[current_x, current_y]

                            # Add neighboring pixels to the queue
                            for x in range(- half_window, half_window + 1):
                                for y in range(- half_window, half_window + 1):
                                    if 0 <= current_x + x < gray_image.shape[0] and 0 <= current_y + y < gray_image.shape[1]:
                                        queue.append((current_x + x, current_y + y))

        return self.visualize_regions(segmented)

    def visualize_regions(self, image):
        """
        Visualizes the regions in the given image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The segmented image with the regions visualized.
        """
        contours, _ = cv2.findContours(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on input image
        output_image = self.image.copy()
        segmented_img = cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)
        return segmented_img


#TODO: link here 
def kmeans_segment_(image, k=5, max_iters=100, save_path='segmented_image.png'):
    """
    Segments an image using K-means clustering on the pixel values.

    Args:
        image: input image.
        K (int): Number of desired clusters.
        max_iters (int): Maximum number of iterations for the K-means algorithm.
        save_path (str): Path where the segmented image will be saved.

    Returns:
        np.ndarray: An array representing the segmented image.
    """

    # Flatten image
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Clustering
    kmeans = KMeans(K=k, max_iters=max_iters)
    labels = kmeans.predict(pixel_values)
    
    # Reshape labels and convert to int type
    labels = labels.astype(int)
    segmented_image = labels.reshape(image.shape[:-1])
    
    # Map clusters to original image colors (average color of the cluster)
    masked_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(k):
        masked_image[segmented_image == i] = kmeans.centroids[i]

    # Convert back to uint8 and save
    masked_image = np.uint8(masked_image)
    plt.imsave(save_path, masked_image)
    
    return masked_image

# def main():

#     X,y = make_classification(100,n_features=2,n_redundant=0)
#     print(X.shape)

#     clusters = Agglomerative_Clustering(linkage_type='complete')
#     Z = clusters.fit(X)
#     Labels = clusters.predict(num_cluster=3)

#     clustering = AgglomerativeClustering(n_clusters=3,linkage='complete').fit(X)
#     skLabel = clustering.labels_

#     fig, ax = plt.subplots(2,2,facecolor='white',figsize=(15,5*2),dpi=120)

#     # Cluster
#     for i in range(3):
#         myIndices = Labels==i
#         skIndices = skLabel==i
#         ax[0,0].scatter(x=X[myIndices,0], y=X[myIndices,1],label=i)
#         ax[0,1].scatter(x=X[skIndices,0], y=X[skIndices,1],label=i)
        
#     ax[0,0].set_title('Custom | Cluster')
#     ax[0,1].set_title('Sklearn | Cluster')
#     ax[0,0].legend()
#     ax[0,1].legend()

#     # Dendrogram
#     z = hierarchy.linkage(X, 'complete') # scipy agglomerative cluster
#     hierarchy.dendrogram(Z, ax=ax[1,0]) # plotting mine with their function
#     hierarchy.dendrogram(z, ax=ax[1,1]) # plotting their with their function

#     ax[1,0].set_title('Custom | Dendrogram')
#     ax[1,1].set_title('Sklearn | Dendrogram')
#     plt.show()



# if __name__ == '__main__':
#     img = cv2.imread("images/shore.jpg")
#     im = kmeans_segment_(img, k=2)
#     plt.imshow(im)
#     plt.show()