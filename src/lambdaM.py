import cv2
import numpy as np

class LambdaMin:
    def compute_gradients(self, image):
        """
        Compute the gradients of an image using the Sobel operator.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            ix (numpy.ndarray): The gradient in the x direction.
            iy (numpy.ndarray): The gradient in the y direction.
        """
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ix = cv2.filter2D(image, -1, sobel, borderType=cv2.BORDER_REFLECT101)
        iy = cv2.filter2D(image, -1, sobel.T, borderType=cv2.BORDER_REFLECT101)
        return ix, iy

    def compute_lambda_min(self, ix, iy, window=3):
        """
        Compute the minimum lambda value for each pixel in an image using the Sobel operator.

        Parameters:
            ix (numpy.ndarray): The gradient in the x direction.
            iy (numpy.ndarray): The gradient in the y direction.
            window (int, optional): The size of the window used for blurring. Defaults to 3.

        Returns:
            numpy.ndarray: A matrix containing the minimum lambda value for each pixel.
        """
        ixx = cv2.blur(ix * ix, (window, window))
        iyy = cv2.blur(iy * iy, (window, window))
        ixy = cv2.blur(ix * iy, (window, window))

        lambda_matrix = np.zeros_like(ix)
        for x, y in np.ndindex(ix.shape):
            H = np.array([[ixx[x, y], ixy[x, y]], [ixy[x, y], iyy[x, y]]])
            eigvals = np.linalg.eigvalsh(H)
            lambdamin = eigvals.min(initial=0)
            lambda_matrix[x, y] = lambdamin
        return lambda_matrix

    def detect_corners(self, image, threshold=0.998, window=3):
        """
        Detects corners in an image using the provided parameters.

        Parameters:
            image (numpy.ndarray): The input image.
            threshold (float, optional): The threshold value for corner detection. Defaults to 0.998.
            window (int, optional): The size of the window used for blurring. Defaults to 3.

        Returns:
            numpy.ndarray: The image with detected corners.
        """
        if image is None:
            raise ValueError("Image is Empty")
        
        try:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

            ix, iy = self.compute_gradients(smoothed)

            lambda_min_matrix = self.compute_lambda_min(ix, iy, window)

            threshold_value = np.quantile(np.abs(lambda_min_matrix), threshold)
            corners = np.argwhere(np.abs(lambda_min_matrix) > threshold_value)

            image_with_corners = image.copy()
            for corner in corners:
                cv2.circle(image_with_corners, (corner[1], corner[0]), 2, (0, 255, 0), -1)

            return image_with_corners
        
        except Exception as e:
            print(str(e))
            return False

