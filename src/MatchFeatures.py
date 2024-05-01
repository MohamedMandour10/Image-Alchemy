import cv2
import numpy as np
from src.SIFT import SIFT_Detector

class MatchFeatures():
    def detect_keypoints_and_descriptors(self, image):
        """
        Detects keypoints and descriptors in the given image using the SIFT algorithm.

        Parameters:
            image (numpy.ndarray): The input image in which keypoints and descriptors will be detected.

        Returns:
            tuple: A tuple containing two lists. The first list contains keypoints with assigned orientations, and the second list contains descriptors.
        """
        sift_detector = SIFT_Detector(image.copy())
        return sift_detector.compute_keypoints_and_descriptors()

    def ssd(self, descriptor1, descriptor2):
        """
        Computes the Sum of Squared Differences (SSD) between two descriptors.

        Parameters:
            descriptor1 (numpy.ndarray): The first descriptor.
            descriptor2 (numpy.ndarray): The second descriptor.

        Returns:
            float: The SSD value between the two descriptors.
        """
        return np.sum((descriptor1 - descriptor2) ** 2)

    def ncc(self, descriptor1, descriptor2):
        """
        Calculate the normalized cross-correlation (NCC) between two descriptors.

        Parameters:
            descriptor1 (numpy.ndarray): The first descriptor.
            descriptor2 (numpy.ndarray): The second descriptor.

        Returns:
            float: The NCC value between the two descriptors.

        Normalized Cross-Correlation (NCC) is a measure of similarity between two vectors.
        It is calculated by first normalizing the descriptors by subtracting the mean and dividing by the standard deviation.
        Then, the dot product of the normalized descriptors is calculated, which represents the correlation between the vectors.
        The NCC value ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).
        """
        descriptor1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1) + 1e-8)
        descriptor2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2) + 1e-8)
        return np.dot(descriptor1_normalized, descriptor2_normalized)

    def match_keypoints(self, des1, des2, method='ssd', threshold=0.75):
        """
        Matches keypoints between two sets of descriptors using a specified method and threshold.

        Parameters:
            des1 (list): A list of descriptors for the first set of keypoints.
            des2 (list): A list of descriptors for the second set of keypoints.
            method (str, optional): The method to use for matching descriptors. Defaults to 'ssd'.
            threshold (float, optional): The threshold for matching descriptors. Defaults to 0.75.

        Returns:
            list: A list of tuples containing the indices and scores of matched keypoints.
        """
        matches = []
        for i, descriptor1 in enumerate(des1):
            best_match_index = -1
            best_match_score = float('inf') if method == 'ssd' else -float('inf')
            for j, descriptor2 in enumerate(des2):
                if method == 'ssd':
                    score = self.ssd(descriptor1, descriptor2)
                    if score < best_match_score:
                        best_match_index = j
                        best_match_score = score
                elif method == 'ncc':
                    score = self.ncc(descriptor1, descriptor2)
                    if score > best_match_score:
                        best_match_index = j
                        best_match_score = score
            if (method == 'ssd' and best_match_score < threshold) or (method == 'ncc' and best_match_score > threshold):
                matches.append((i, best_match_index, best_match_score))
        return matches

    def get_keypoint_coordinates(self, kp1, kp2, matches):
        """
        Get the coordinates of keypoints from two sets of keypoints.

        Parameters:
            kp1 (list): A list of keypoints from the first set.
            kp2 (list): A list of keypoints from the second set.
            matches (list): A list of matches between keypoints from the first and second sets.

        Returns:
            tuple: A tuple containing two numpy arrays. The first array contains the coordinates of the keypoints from the first set, and the second array contains the coordinates of the keypoints from the second set.
        """
        points1 = np.array([(kp1[m[1]][1],kp1[m[1]][0]) for m in matches])
        points2 = np.array([(kp2[m[1]][1],kp2[m[1]][0]) for m in matches])
        return points1, points2

    def draw_matches(self, image1, image2, points1, points2):
        """
        Draws matches between two images based on the given keypoints.
        
        Parameters:
            image1 (numpy.ndarray): The first input image.
            image2 (numpy.ndarray): The second input image.
            points1 (numpy.ndarray): The keypoints from the first image.
            points2 (numpy.ndarray): The keypoints from the second image.
        
        Returns:
            numpy.ndarray: An image with the matches drawn between the keypoints.
        """
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)

        matched_image[:image1.shape[0], :image1.shape[1]] = image1
        matched_image[:image2.shape[0], image1.shape[1]:] = image2

        # Generate random colors for drawing lines
        np.random.seed(42)  # For reproducibility
        colors = np.random.randint(0, 255, (len(points1), 3))

        for i in range(len(points1)):
            pt1 = (int(points1[i][0]), int(points1[i][1]))
            pt2 = (int(points2[i][0]) + image1.shape[1], int(points2[i][1]))
            color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            cv2.line(matched_image, pt1, pt2, color, 1)

            # Draw a circle at the beginning and end points
            cv2.circle(matched_image, pt1, 3, color, -1)  # Beginning point
            cv2.circle(matched_image, pt2, 3, color, -1)  # End point
        cv2.imwrite("/imgs/matched_image.jpg",matched_image)
        return matched_image
    
    def match(self, image1, image2, method = 'ssd'):
        """
        Matches two images based on their keypoints using the specified method.

        Parameters:
            image1 (numpy.ndarray): The first input image.
            image2 (numpy.ndarray): The second input image.
            method (str, optional): The method to use for matching keypoints. Defaults to 'ssd'.

        Returns:
            numpy.ndarray: An image with the matched keypoints drawn between the two images.
        """

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detect_keypoints_and_descriptors(gray1)
        kp2, des2 = self.detect_keypoints_and_descriptors(gray2)
        
        matches = self.match_keypoints(des1, des2, method=method, threshold=10000 if method == 'ssd' else 0.8)

        points1, points2 = self.get_keypoint_coordinates(kp1, kp2, matches)

        matched_image = self.draw_matches(image1, image2, points1, points2)

        return matched_image