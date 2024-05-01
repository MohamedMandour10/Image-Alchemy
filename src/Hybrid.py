import numpy as np
from src.Filters import Filter
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift

class Hybrid:
    def __init__(self):
        self.filtered_img_one = None
        self.filtered_img_two = None

    def low_pass(self, image, smoothing_degree):
         fft_img = fftshift(fft2(image))
         rows, cols = image.shape
         crow, ccol = rows // 2, cols // 2
         cutoff_frequency = 10
         smoothing_degree = (smoothing_degree + 1) / 25.6
         mask = np.zeros((rows, cols), np.float32)
         for i in range(rows):
             for j in range(cols):
                 distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                 mask[i, j] = np.exp(-0.5 * (distance / cutoff_frequency) ** smoothing_degree)
         low_pass_fft = fft_img * mask
         low_pass = ifft2(ifftshift(low_pass_fft))
         self.filtered_img_one = low_pass
         low_pass = np.abs(low_pass)
         low_pass = normalize_image(low_pass)
         low_pass = low_pass.astype(np.uint8)
         return low_pass
    
    def high_pass(self, image, edge_degree):
         fft_img = fftshift(fft2(image))
         rows, cols = image.shape
         crow, ccol = rows // 2, cols // 2
         cutoff_frequency = 10
         edge_degree = (edge_degree + 1) / 25.6 
         mask = np.zeros((rows, cols), np.float32)
         for i in range(rows):
             for j in range(cols):
                 distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                 mask[i, j] = 1 - np.exp(-0.5 * (distance / cutoff_frequency) ** edge_degree)
         high_pass_fft = fft_img * mask
         high_pass = ifft2(ifftshift(high_pass_fft))
         self.filtered_img_two = high_pass
         high_pass = np.abs(high_pass)
         high_pass = normalize_image(high_pass)
         high_pass = high_pass.astype(np.uint8)
         return high_pass

    def generate_hybrid(self):
         img1 = self.filtered_img_one
         img2 = self.filtered_img_two
         if img1 is None or img2 is None:
             return
         if img2.shape != img1.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = resize_complex_array(img1, (min_width, min_height))
            img2 = resize_complex_array(img2, (min_width, min_height))
         hybrid_image = np.abs(img1 + img2)
         
         hybrid_image = normalize_image(hybrid_image)
         hybrid_image = hybrid_image.astype(np.uint8)
         return hybrid_image
    

def normalize_image(image):
      min_val = np.min(image)
      max_val = np.max(image)

      # Check if min_val and max_val are equal
      if min_val == max_val:
          return image

      # Perform normalization
      normalized_image = ((image - min_val) / (max_val - min_val)) * 255
      normalized_image = normalized_image.astype(np.uint8)

      return normalized_image

def resize_complex_array(complex_array, new_shape):
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    resized_real_part = cv2.resize(real_part, new_shape[::-1])
    resized_imag_part = cv2.resize(imag_part, new_shape[::-1])
    resized_complex_array = resized_real_part + 1j * resized_imag_part
    
    return resized_complex_array