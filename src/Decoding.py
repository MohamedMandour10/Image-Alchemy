import numpy as np
import cv2

class Decoding:
    def __init__(self, original_image):
        self.gray = cv2.convertScaleAbs(original_image)

    def equalize(self):
        # Calculate the histogram of the original image
        histogram, _ = np.histogram(self.gray.flatten(), bins=256, range=[0, 256])

        # Calculate the cumulative distribution function (CDF) of the histogram
        cdf = histogram.cumsum()

        # Normalize the CDF to the range [0, 255]
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

        # Interpolate the values of the CDF for each pixel in the image
        equalized_image = np.interp(self.gray.flatten(), range(256), cdf_normalized).reshape(self.gray.shape).astype(np.uint8)
        return equalized_image

    def normalize(self):
        # Convert the image to floating-point format
        image_float = self.gray.astype(np.float32)

        # Determine the minimum and maximum pixel values
        min_val = np.min(image_float)
        max_val = np.max(image_float)

        # Normalize the image to the range [0, 1]
        normalized_image = 255 * (image_float - min_val) / (max_val - min_val)

        return normalized_image.astype(np.uint8)
