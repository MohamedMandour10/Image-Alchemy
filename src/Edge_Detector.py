import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, original_img):
        self.gray = cv2.convertScaleAbs(original_img)

    def sobel_detector(self):
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Convolve the image with the kernels
        gradient_x = cv2.filter2D(self.gray, cv2.CV_64F, sobel_x)
        gradient_y = cv2.filter2D(self.gray, cv2.CV_64F, sobel_y)

        # Compute gradient magnitude
        gradient_magnitude = cv2.sqrt(cv2.pow(gradient_x, 2), cv2.pow(gradient_y, 2))

        gradient_magnitude *= 255.0 / gradient_magnitude.max()

        return gradient_magnitude.astype(np.uint8)

    def roberts_detector(self):
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])

        roberts_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
        roberts = cv2.sqrt(cv2.pow(roberts_x, 2), cv2.pow(roberts_y, 2))

        return roberts.astype(np.uint8)

    def canny_detector(self):
        # Step 2: Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(self.gray, (5, 5), 0)
        low_threshold = 5
        high_threshold = 20
        # Step 3: Compute gradient magnitude and direction
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

        # Step 4: Non-maximum suppression
        suppressed_image = np.zeros_like(gradient_magnitude)
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                angle = gradient_direction[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]

        # Step 5: Hysteresis thresholding
        edge_image = np.zeros_like(suppressed_image)
        weak_edges = (suppressed_image > low_threshold) & (suppressed_image <= high_threshold)
        strong_edges = suppressed_image > high_threshold
        edge_image[strong_edges] = 255
        for i in range(1, edge_image.shape[0] - 1):
            for j in range(1, edge_image.shape[1] - 1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        edge_image[i, j] = 255

        return edge_image.astype(np.uint8)

    def prewitt_detector(self):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
        prewitt = cv2.sqrt(cv2.pow(prewitt_x, 2), cv2.pow(prewitt_y, 2))

        return prewitt.astype(np.uint8)
