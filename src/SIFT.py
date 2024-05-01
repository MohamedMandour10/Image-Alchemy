import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class SIFT_Detector:
    def __init__(self, image):
        self.image = image
        self.gaussian_pyramids = None
        self.grad_x = None
        self.grad_y = None

    def generate_gaussian_pyramid(self, num_octaves= 4, num_scales= 5):
        """Generates a Gaussian pyramid of images

        Args:
            image (numpy.ndarray): The base image
            num_octaves (int): The number of octaves (number of downsamples)
            num_scales (int): The number of scales per octave (number of blurred images per octave)

        Returns:
            list: A list of lists of images. Each inner list represents an octave,
            and each image in that list is a single blurred image in that octave
        """
        image = self.image.copy().astype("float64")
        gaussian_pyramid = []

        for octave in range(num_octaves):

            # Octave images will be a list of images
            octave_images = []

            for scale in range(num_scales):
                # Calculate the sigma for this blur
                sigma = 2 ** octave * (2 ** (scale / num_scales)) # 2 * (2 ** (1 / 5)) first iteration, 2 * (2 ** (2 / 5)) second iteration
                # Apply Gaussian blur to the image with the calculated sigma
                octave_images.append(cv2.GaussianBlur(image, (0, 0), sigma))
 
            gaussian_pyramid.append(octave_images)
            # Downsample the image by a factor of 2 for the next octave
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
            self.gaussian_pyramids = gaussian_pyramid
        return gaussian_pyramid

    def generate_difference_of_gaussians(self, gaussian_pyramid):
        """Generates a Difference of Gaussians (DoG) pyramid of images

        The DoG pyramid is a pyramid of images, where each image is the difference
        between two successive blurred versions of the base image. This is used
        to detect scale-invariant features.

        Args:
            gaussian_pyramid (list): A list of lists of images. Each inner list
                represents an octave (a downsampled version of the image), and each
                image in that list is a single blurred version of the base image
                in that octave.

        Returns:
            list: A list of lists of images. Each inner list represents an octave,
                and each image in that list is a single DoG image in that octave.
        """
        dog_pyramid = []

        # Iterate over each octave 
        for octave_images in gaussian_pyramid:
            dog_octave = []  

            # Calculate the difference between successive blurred versions of the 
            # image in this octave
            for i in range(len(octave_images) - 1):
                difference = octave_images[i + 1] - octave_images[i]
                dog_octave.append(difference)

            dog_pyramid.append(dog_octave)

        return dog_pyramid

    def find_local_extrema(self, prev_image, next_image, pixel_value, x, y):
        """
        Checks whether the pixel at (x, y) in the image is a local extrema
        by comparing it to its 26 neighboring pixels.

        A local extrema is a pixel that is either a maximum or minimum in its
        neighborhood.

        Args:
            prev_image (numpy.ndarray): The image to which the current image has been blurred
            next_image (numpy.ndarray): The image to which the current image has not been blurred
            pixel_value (float): The grayscale value of the current pixel
            x (int): The x-coordinate of the current pixel
            y (int): The y-coordinate of the current pixel

        Returns:
            bool: True if the current pixel is a local extrema, False otherwise
        """

        # Neighboring pixels around the current pixel
        #   (-1,-1) | (0,-1) | (1,-1)
        #   ---------------------------
        #   (-1, 0) | (0, 0) | (1, 0)
        #   ---------------------------
        #   (-1, 1) | (0, 1) | (1, 1)
        # Check against 26 neighboring pixels
        return all(prev_image[y + indices_i, x + indices_j] < pixel_value for indices_i in range(-1, 2) for indices_j in
                  range(-1, 2)) and \
            all(pixel_value > next_image[y + indices_i, x + indices_j] for indices_i in range(-1, 2) for indices_j in
                range(-1, 2))


    def eliminate_edges(self, image, y, x):
        """
        Compute the second-order derivatives and check if the current pixel is a well-localized edge using the
        Hessian matrix.

        The Hessian matrix is defined as:
            [Ixx Ixy]
            [Ixy Iyy]

        where Ixx = dI/dx^2, Ixy = dI/dxdy, Iyy = dI/dydy.
        The trace of the Hessian matrix is tr(H) = Ixx + Iyy and its determinant is det(H) = IxxIyy - Ixy^2.

        The eigenvalue ratio is defined as r = (tr(H)^2)/det(H). If r > (r_th + 1)^2/r_th, the pixel is considered
        a well-localized edge.

        Args:
            image (numpy.ndarray): The image to analyze
            y (int): The y-coordinate of the current pixel
            x (int): The x-coordinate of the current pixel

        Returns:
            bool: True if the current pixel is a well-localized edge, False otherwise
        """

        # Compute second-order derivatives with appropriate data type
        Ixx = np.float64(image[y, x + 1]) + np.float64(image[y, x - 1]) - 2 * np.float64(image[y, x])
        Iyy = np.float64(image[y + 1, x]) + np.float64(image[y - 1, x]) - 2 * np.float64(image[y, x])
        Ixy = (np.float64(image[y + 1, x + 1]) - np.float64(image[y + 1, x - 1]) - np.float64(image[y - 1, x + 1]) +
               np.float64(image[y - 1, x - 1])) / 4

        # Compute Trace and Determinant of the Hessian matrix
        trace_H = Ixx + Iyy
        det_H = Ixx * Iyy - Ixy ** 2

        # Compute the eigenvalue ratio with a small epsilon to overcome division by zero
        epsilon = 10e-6
        response = (trace_H ** 2) / (det_H + epsilon)

        # Define the threshold eigenvalue ratio
        r_th = 10
        return response > ((r_th + 1) ** 2) / r_th

    def find_keypoints(self, dog_pyramid, threshold=0.03, intensity_threshold=8):
        """
        Finds local extrema in the given DoG pyramid and returns a list of tuples representing the keypoints.
        
        Args:
            dog_pyramid (list): A list of lists of images representing the DoG pyramid.
            threshold (float, optional): The threshold value for contrast. Defaults to 0.03.
            intensity_threshold (int, optional): The threshold value for pixel intensity. Defaults to 8.
        
        Returns:
            list: A list of tuples representing the keypoints. Each tuple contains the y-coordinate,
            x-coordinate, and octave index of a local extremum.
        """
        local_extrema = []  # List to store local extrema

        # Iterate over each octave in the DoG pyramid
        for octave_idx, octave_images in enumerate(dog_pyramid):
            # Ignore the first and last DoG images in the octave
            for image_idx, image in enumerate(octave_images[1:-1], start=1):
                # Get the previous and next images in the octave
                prev_image = octave_images[image_idx - 1]
                next_image = octave_images[image_idx + 1]

                # Iterate over each pixel in the image
                for y in range(1, image.shape[0] - 1):
                    for x in range(1, image.shape[1] - 1):
                        pixel_value = image[y, x]  # Get the value of the current pixel

                        # Check if the pixel value is a local extremum
                        if self.find_local_extrema(prev_image, next_image, pixel_value, x, y):

                            # Calculate the second-order Taylor expansion value
                            taylor_expansion = np.float64(pixel_value) + 0.5 * (np.float64(prev_image[y, x]) +
                                                                                np.float64(
                                                                                    next_image[y, x]) - 2 * np.float64(
                                        pixel_value))

                            # Discard keypoints with low contrast
                            if abs(taylor_expansion) >= threshold and pixel_value >= intensity_threshold:
                                try:
                                    # Check if the keypoint is poorly localized and has a high edge response
                                    if self.eliminate_edges(image, y, x):
                                        continue  # Skip this keypoint

                                    # Add the keypoint to the list of local extrema
                                    local_extrema.append((y, x, octave_idx))
                                except Exception as e:
                                    print(f"Error processing pixel ({x}, {y}): {e}")
                                    continue

        return local_extrema


    def assign_orientation(self, keypoints, DOG):
        """
        Assigns orientation to keypoints based on their gradients and creates histograms to find dominant orientation.
        
        Args:
            keypoints (list): List of keypoints, each containing y-coordinate, x-coordinate, and octave index.
            DOG (list): List of Difference of Gaussian images for each octave.
        
        Returns:
            list: A list of keypoints with assigned orientations.
        """
        keypoints_with_orientation = []
        for keypoint in keypoints:
            y, x, octave_idx = keypoint  # Unpack the tuple
            octave_images = DOG[octave_idx]
            image = octave_images[0]
            size = octave_images[1]

            # Compute gradients in the local neighborhood of the keypoint
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate magnitude and orientation
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)

            # Create histogram
            hist = np.zeros(36)
            for i in range(-8, 9):
                for j in range(-8, 9):
                    if 0 <= y + i < image.shape[0] and 0 <= x + j < image.shape[1]:
                        angle = np.rad2deg(orientation[y + i, x + j])
                        bin_idx = int((angle + 180) / 10)
                        hist[bin_idx % 36] += magnitude[y + i, x + j]

            # Find dominant orientation
            dominant_orientation = np.argmax(hist) * 10

            # Append tuple of (y, x, octave, orientation) to keypoints_with_orientation list
            keypoints_with_orientation.append((y, x, octave_idx, dominant_orientation))

            # Check for additional keypoints
            for i in range(len(hist)):
                if i == np.argmax(hist):
                    continue
                if hist[i] >= 0.8 * np.max(hist):  # Threshold for significant peaks
                    keypoints_with_orientation.append(
                        (y, x, octave_idx, i * 10))  # Create a new keypoint tuple with the adjusted orientation

        return keypoints_with_orientation


    def compute_descriptor(self, keypoint, images):
        """
        Compute the descriptor for a given keypoint in an image.

        Parameters:
            keypoint (tuple): The keypoint coordinates (y, x, octave_idx, orientation).
            images (list): The list of images in the octave.

        Returns:
            list: The computed descriptor for the keypoint.
        """
        y, x, octave_idx, orientation = keypoint
        octave_images = images[octave_idx]
        image = octave_images[0]

        # Get the 16x16 neighborhood around the keypoint
        patch = image[y - 8:y + 8, x - 8:x + 8]

        # Rotate the patch according to the orientation
        rot_mat = cv2.getRotationMatrix2D((8, 8), -orientation, 1.0)
        rotated_patch = cv2.warpAffine(patch, rot_mat, (16, 16))

        # Divide the patch into 4x4 sub-blocks
        sub_blocks = [rotated_patch[i:i + 4, j:j + 4] for i in range(0, 16, 4) for j in range(0, 16, 4)]

        descriptor = []

        for sub_block in sub_blocks:
            # Compute gradients in x and y directions
            grad_x = cv2.Sobel(sub_block, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(sub_block, cv2.CV_64F, 0, 1, ksize=3)

            # Compute magnitude and orientation
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)

            # Create histogram for orientation
            hist = np.zeros(8)
            for i in range(4):
                for j in range(4):
                    angle = np.rad2deg(orientation[i, j])
                    bin_idx = int(angle / 45)  # Divide the angle range into 8 bins
                    hist[bin_idx] += magnitude[i, j]

            # Append histogram to descriptor
            descriptor.extend(hist)

        # Normalize descriptor
        descriptor /= np.linalg.norm(descriptor)

        return descriptor

    def compute_descriptors(self, keypoints_with_orientation, images):
        """
        Compute descriptors for a list of keypoints with orientation using a list of images.

        Parameters:
            keypoints_with_orientation (list): A list of tuples representing keypoints with orientation.
                Each tuple contains the y-coordinate, x-coordinate, octave index, and orientation.
            images (list): A list of images in the octave.

        Returns:
            list: A list of descriptors computed for each keypoint.
        """
        descriptors = []
        for keypoint in keypoints_with_orientation:
            descriptor = self.compute_descriptor(keypoint, images)
            descriptors.append(descriptor)
        return descriptors

    def compute_keypoints_and_descriptors(self, sigma=1.6, num_octaves=4, num_scales=5):
        """
        Computes keypoints and descriptors based on the given parameters.

        Args:
            sigma (float, optional): The standard deviation of the Gaussian kernel used for blurring. Defaults to 1.6.
            num_octaves (int, optional): The number of octaves (number of downsamples) in the Gaussian pyramid. Defaults to 4.
            num_scales (int, optional): The number of scales per octave (number of blurred images per octave) in the Gaussian pyramid. Defaults to 5.

        Returns:
            tuple: A tuple containing two lists. The first list contains keypoints with assigned orientations, and the second list contains descriptors.
        """

        gaussian_pyramid = self.generate_gaussian_pyramid(num_octaves, num_scales)
        DOG = self.generate_difference_of_gaussians(gaussian_pyramid)
        keypoints = self.find_keypoints(DOG)
        keypoints_after_orientation = self.assign_orientation(keypoints, self.gaussian_pyramids)
        descriptors = self.compute_descriptors(keypoints_after_orientation, self.gaussian_pyramids)

        return keypoints_after_orientation, descriptors
