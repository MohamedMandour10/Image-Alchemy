import cv2
import numpy as np

class Noise:
    def __init__(self, original_img):
        self.original_img = original_img
        self.noisy_img = None

    def slat_and_pepper(self, density=0.05):
        """
        Adds salt and pepper noise to the grayscale image.
        
        Parameters:
            density (float): Density of the salt and pepper noise. Default is 0.05.
        """
        # Create a copy of the original image
        self.noisy_img = self.original_img.copy()

        # Calculate the number of salt pixels based on the density
        num_salt = np.ceil(density * self.original_img.size * 0.5)

        # Generate random coordinates for salt pixels
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in self.original_img.shape]

        # Add salt pixels to the noisy image
        self.noisy_img[coords_salt[0], coords_salt[1]] = 255

        # Calculate the number of pepper pixels based on the density
        num_pepper = np.ceil(density * self.original_img.size * 0.5)

        # Generate random coordinates for pepper pixels
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.original_img.shape]

        # Add pepper pixels to the noisy image
        self.noisy_img[coords_pepper[0], coords_pepper[1]] = 0

        # Return the noisy image as unsigned 8-bit integers
        return self.noisy_img.astype(np.uint8)

    def gaussian_noise(self, mean=0, stddev=25):
        """
        Adds Gaussian noise to the image.
        
        Parameters:
            mean (float): Mean of the Gaussian distribution. Default is 0.
            stddev (float): Standard deviation of the Gaussian distribution. Default is 25.
        """
        # Make a copy of the original image
        self.noisy_img = self.original_img.copy()

        # Add Gaussian noise to the image
        noise = np.random.normal(mean, stddev, self.original_img.shape)
        self.noisy_img = np.clip(self.noisy_img + noise, 0, 255)

        # Convert the noisy image to unsigned 8-bit integer
        return self.noisy_img.astype(np.uint8)

    def uniform_noise(self, min_val=0, max_val=100):
        """
        Adds uniform noise to the image.
        
        Parameters:
            min_val (int): Minimum value of the uniform distribution. Default is 0.
            max_val (int): Maximum value of the uniform distribution. Default is 255.
        """
        # Create a copy of the original image
        self.noisy_img = self.original_img.copy()

        # Generate random noise with values between min_val and max_val
        noise = np.random.randint(min_val, max_val + 1, self.original_img.shape)

        # Add the noise to the noisy image and ensure the pixel values stay within the range [0, 255]
        self.noisy_img = np.clip(self.noisy_img + noise, 0, 255)

        # Convert the noisy image to unsigned 8-bit integer type and return it
        return self.noisy_img.astype(np.uint8)