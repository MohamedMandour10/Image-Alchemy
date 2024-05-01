import cv2
import numpy as np

class Thresholding:
    def __init__(self, image):
        self.image = image

    def __optimal_threshold(
        self, hist: np.ndarray, initial_threshold: int = 120, max_iter: int = 100, min_diff: float = 1e-5) -> int:
        """
        Perform an optimal threshold algorithm on a histogram.

        Parameters:
            hist (np.ndarray[int]): The input histogram.
            initial_threshold (int): The initial threshold value.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            min_diff (float, optional): The tolerance for convergence. Defaults to 1e-5.

        Returns:
            int: The optimal threshold value.
        """
        # Normalize the histogram
        hist_norm = hist / hist.sum()

        # Initialize threshold
        t = initial_threshold

        for _ in range(max_iter):
            # Cut the distribution into two parts
            h1 = hist_norm[:t]
            h2 = hist_norm[t:]

            # Compute centroids, mean of the two parts
            m1 = (np.arange(t, dtype=np.float64) * h1).sum() / (h1.sum() + 1e-5)
            m2 = (np.arange(t, 256, dtype=np.float64) * h2).sum() / (h2.sum() + 1e-5)

            # Compute new threshold
            t_new = int(round((m1 + m2) / 2))

            # Check convergence
            if abs(t_new - t) < min_diff:
                break

            t = t_new

        return t

    def optimal_threshold_global(
            self, initial_threshold: int = 125, max_iter: int = 100, min_diff: float = 1e-5) -> tuple[int, np.ndarray]:
        """
        Calculates the optimal threshold value for image segmentation using the optimal thresholding algorithm.

        Parameters:
            initial_threshold (int): The initial threshold value. Default is 125.
            max_iter (int): The maximum number of iterations for the algorithm. Default is 100.
            min_diff (float): The tolerance for convergence. Default is 1e-5.

        Returns:
            tuple: A tuple containing the optimal threshold value (int) and the thresholded image (numpy array).
        """
        # Convert the image to grayscale if it's not already
        if self.image.ndim == 3:
            gray: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # Compute the histogram
        hist: np.ndarray = cv2.calcHist([gray], [0], None, [256], [0,256])

        # Reshape the histogram to 1D array
        hist = hist.flatten()

        optimal_t: int = self.__optimal_threshold(hist, initial_threshold, max_iter, min_diff)

        _, thresholded_image = cv2.threshold(gray, optimal_t, 255, cv2.THRESH_BINARY)

        return optimal_t, thresholded_image
    

    def optimal_threshold_local(self, block_size=10):
        """
        Applies local optimal thresholding to the image.

        Parameters:
            block_size (int): Size of the local region for thresholding. Default is 10.

        Returns:
            np.ndarray: The thresholded image.
        """
        # Convert the image to grayscale if it's not already
        if self.image.ndim == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        height, width = gray_image.shape
        thresholded_image = np.zeros_like(gray_image)

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Extract local region
                region = gray_image[i:i+block_size, j:j+block_size]
                # Compute the histogram
                hist: np.ndarray = cv2.calcHist([region], [0], None, [256], [0,256])
                # Reshape the histogram to 1D array
                hist = hist.flatten()
                # Calculate optimal threshold for the region
                optimal_t = self.__optimal_threshold(hist)
                # Apply thresholding to the region
                _, thresholded_region = cv2.threshold(region , optimal_t, 255, cv2.THRESH_BINARY)
                # Place thresholded region back into the image
                thresholded_image[i:i+block_size, j:j+block_size] = thresholded_region

        return thresholded_image


    #TODO:LINK THIS FOR GLOBAL THRESHOLDING OTSU
    def otsu_threshold_global(self):
        """
        Applies Otsu's thresholding algorithm to find the optimal threshold.

        Args:
            self.image (np.ndarray): Input grayscale image.

        Returns:
            int: The optimal threshold value.
        """
        thresholds = []
        for t in range(self.image.min() + 1, self.image.max()):
            below_threshold = self.image[self.image < t]
            above_threshold = self.image[self.image >= t]
            Wb = len(below_threshold) / (self.image.shape[0] * self.image.shape[1])
            Wa = len(above_threshold) / (self.image.shape[0] * self.image.shape[1])
            var_b = np.var(below_threshold)
            var_a = np.var(above_threshold)
            thresholds.append(Wb * var_b + var_a * Wa)
        try:
            min_threshold = min(thresholds)
            optimal_threshold = thresholds.index(min_threshold)
        except ValueError:
            optimal_threshold = 0

        thresholded_image = np.where(self.image >= self.image.min() + optimal_threshold, 255, 0).astype(np.uint8)
    
        return  thresholded_image
    

    def apply_otsu_on_region(self, img):
        """
        Applies Otsu's thresholding algorithm to find the optimal threshold.

        Args:
            self.image (np.ndarray): Input grayscale image.

        Returns:
            int: The optimal threshold value.
        """
        thresholds = []
        for t in range(img.min() + 1, img.max()):
            below_threshold = img[img < t]
            above_threshold = img[img >= t]
            Wb = len(below_threshold) / (img.shape[0] * img.shape[1])
            Wa = len(above_threshold) / (img.shape[0] * img.shape[1])
            var_b = np.var(below_threshold)
            var_a = np.var(above_threshold)
            thresholds.append(Wb * var_b + var_a * Wa)
        try:
            min_threshold = min(thresholds)
            optimal_threshold = thresholds.index(min_threshold)
        except ValueError:
            optimal_threshold = 0

        thresholded_image = np.where(img >= img.min() + optimal_threshold, 255, 0).astype(np.uint8)
    
        return optimal_threshold , thresholded_image
    
    #TODO: LINK THIS FOR LOCAL THRESHOLD OTSU
    def otsu_threshold_local(self, n_regions=4):
        """
        Applies Otsu's thresholding algorithm to find the optimal threshold.

        Args:
            self.image (np.ndarray): Input grayscale image.

        Returns:
            int: The optimal threshold value.
        """

        # devide the image into n_regions of equal images
        imgs_slices = []
        for i in range(n_regions):
            start_row = (self.image.shape[0] // n_regions) * i
            end_row = (self.image.shape[0] // n_regions) * (i + 1)
            region = self.image[start_row:end_row, :]
            imgs_slices.append(region)
        
        for idx, img_slice in enumerate(imgs_slices):
            _, thresholded_image = self.apply_otsu_on_region(img_slice)
            imgs_slices[idx] = thresholded_image
            
        final_image = np.vstack(imgs_slices)

        return final_image

        
        
    def spectral_threshold_global(self):
        gray_image = self.image

        hist = np.zeros(256)
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                hist[gray_image[i, j]] += 1

        hist_cdf = np.cumsum(hist)
        total_pixels = hist_cdf[-1]
        threshold_low = np.argmax(hist_cdf >= total_pixels / 3)
        threshold_high= np.argmax(hist_cdf >= total_pixels * 2 / 3)

        global_image  = np.zeros_like(gray_image)
        global_image [gray_image <= threshold_low] = 0
        global_image [(gray_image > threshold_low) & (gray_image <= threshold_high)] = 127
        global_image [gray_image > threshold_high] = 255
        

        return global_image
    
    def spectral_threshold_local(self, block_size=20, bandwidth=5):
        gray_image = self.image

        local_image = np.zeros_like(gray_image)

        pad = block_size // 2
        
        for i in range(pad, gray_image.shape[0] - pad):
            for j in range(pad, gray_image.shape[1] - pad):
                block = gray_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                block_mean = np.mean(block)
                threshold_low = block_mean - bandwidth
                threshold_high = block_mean + bandwidth
                if gray_image[i, j] <= threshold_low:
                    local_image[i, j] = 0
                elif gray_image[i, j] > threshold_low and gray_image[i, j] <= threshold_high:
                    local_image[i, j] = 127
                else:
                    local_image[i, j] = 255

        return local_image



# img = cv2.imread("images/cameraman.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (256, 256))
# thres = Thresholding(img)
# im = thres.otsu_threshold_global()

# # Display the thresholded image using cv2.imshow()
# cv2.imshow('Thresholded Image', im)
# cv2.waitKey(0)

# im = thres.otsu_threshold_local()

# # Display the thresholded image using cv2.imshow()
# cv2.imshow('Thresholded local Image', im)
# cv2.waitKey(0)



