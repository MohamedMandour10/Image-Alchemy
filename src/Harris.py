import cv2
import numpy as np

class Harris():
    def grey_scale_img(self, img):
        """
        Convert an image from BGR color space to grayscale.

        Parameters:
            img (numpy.ndarray): The input image in BGR color space.

        Returns:
            numpy.ndarray: The grayscale version of the input image.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def apply_smoothing(self,img, kernel=(3,3)):
        """
        Apply smoothing to an image using Gaussian blur.

        Parameters:
            img (numpy.ndarray): The input image to be smoothed.
            kernel (tuple): The size of the kernel used for smoothing. Default is (3,3).

        Returns:
            numpy.ndarray: The smoothed image.
        """
        return cv2.GaussianBlur(img, kernel, 0)
    
    def compute_gradients(self, image):
        """
        Compute the gradients of an image using the Sobel operator.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            sobelx (numpy.ndarray): The gradient in the x direction.
            sobely (numpy.ndarray): The gradient in the y direction.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return sobelx, sobely
    
    def compute_products_of_derivatives(self, sobelx, sobely):
        """
        Compute the products of the derivatives of an image using the Sobel operator.

        Parameters:
            sobelx (numpy.ndarray): The gradient in the x direction.
            sobely (numpy.ndarray): The gradient in the y direction.

        Returns:
            tuple: A tuple containing the squares of the x and y gradients, and the product of the x and y gradients.
        """
        IxIx = sobelx ** 2
        IyIy = sobely ** 2
        IxIy = sobelx * sobely
        return IxIx, IyIy, IxIy
    
    def compute_sums_of_products(self, IxIx, IyIy, IxIy):
        """
        Compute the sums of products of the given parameters.

        Parameters:
            IxIx (numpy.ndarray): The array containing the squared x derivatives.
            IyIy (numpy.ndarray): The array containing the squared y derivatives.
            IxIy (numpy.ndarray): The array containing the products of x and y derivatives.

        Returns:
            tuple: A tuple containing the sums of the IxIx, IyIy, and IxIy arrays.
        """
        sum_IxIx = self.apply_smoothing(IxIx)
        sum_IyIy = self.apply_smoothing(IyIy)
        sum_IxIy = self.apply_smoothing(IxIy)
        return sum_IxIx, sum_IyIy, sum_IxIy

    def compute_harris_response(self, sum_IxIx, sum_IyIy, sum_IxIy, k):
        """
        Compute the Harris response using the given parameters.

        Parameters:
            sum_IxIx (float): The sum of squared x derivatives.
            sum_IyIy (float): The sum of squared y derivatives.
            sum_IxIy (float): The sum of x and y derivatives products.
            k (float): A constant multiplier.

        Returns:
            float: The computed Harris response value.
        """
        det = (sum_IxIx * sum_IyIy) - (sum_IxIy ** 2)
        trace = sum_IxIx + sum_IyIy
        return det - k * (trace ** 2)

    def threshold_and_get_corners(self, harris_response, threshold):
        """
        Compute the corners in an image based on the Harris response and threshold.

        Parameters:
            harris_response (numpy.ndarray): The Harris response image.
            threshold (float): The threshold value for corner detection.

        Returns:
            numpy.ndarray: The coordinates of the corners in the image.
        """
        corner_threshold = harris_response.max() * threshold
        corners = np.where(harris_response > corner_threshold)
        return np.vstack((corners[1], corners[0])).T

    def draw_corners(self, image, corner_coords):
        """
        Draws circles on the given image at the specified corner coordinates.

        Parameters:
            image (numpy.ndarray): The input image.
            corner_coords (numpy.ndarray): The coordinates of the corners.

        Returns:
            None
        """
        for corner in corner_coords:
            cv2.circle(image, tuple(corner), 2, (0, 255, 0), -1)
        self.write_output_image(image)

    def write_output_image(self, image):
        cv2.imwrite('../imgs/output_harris.jpg', image)

    def harris_corner_detector(self, image, k=0.04, threshold=0.01):
        """
        Applies the Harris corner detection algorithm to an image.

        Parameters:
            image (numpy.ndarray): The input image.
            k (float): A constant multiplier for the Harris response calculation. Default is 0.04.
            threshold (float): The threshold value for corner detection. Default is 0.01.

        Returns:
            numpy.ndarray: The image with circles drawn on the detected corners.
            False: If an error occurs during the corner detection process.
        """
        if image is None:
            raise ValueError("Image is Empty")
        try:
            gray = self.grey_scale_img(image)
            sobelx, sobely = self.compute_gradients(gray)
            IxIx, IyIy, IxIy = self.compute_products_of_derivatives(sobelx, sobely)
            sum_IxIx, sum_IyIy, sum_IxIy = self.compute_sums_of_products(IxIx, IyIy, IxIy)
            harris_response = self.compute_harris_response(sum_IxIx, sum_IyIy, sum_IxIy, k)
            corner_coords = self.threshold_and_get_corners(harris_response, threshold)
            self.draw_corners(image, corner_coords)
            return image
        
        except Exception as e:
            print(str(e))
            return False



