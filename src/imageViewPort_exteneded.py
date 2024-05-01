from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QImage, QColor, QPen
from PyQt6.QtCore import Qt, QPoint
import cv2
import logging
import numpy as np

class ImageViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_img = None
        self.resized_img = None
        self.image_path = None
        self.points = []  # List to store drawing points
        self.drawing_enabled = False  # Flag to enable drawing

    def set_image(self, image_path, grey_flag=False):
        """
        Set the image for the object.

        Args:
            image_path (str): The path to the image file.

        Returns:
            None
        """
        try:
            # Open the image file 
            image = cv2.imread(image_path)

            if image is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")

            self.image_path = image_path
            if not grey_flag:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Set the original_img attribute 
            self.original_img = image
            self.points = []

            self.update_display()

        except FileNotFoundError as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"Error displaying image: {e}")

    def update_display(self):
        """
        Update the display if the original image is not None.
        """
        if self.original_img is not None:
            self.repaint()

    def paintEvent(self, event):
        """
        Override the paint event to draw the image on the widget.

        Args:
        - event: the paint event
        """
        super().paintEvent(event)

        if self.original_img is not None and len(self.original_img.shape) > 1:
            painter_img = QPainter(self)
            self.original_img = self.original_img.astype(np.uint8)  # Convert image to uint8
            height, width = self.original_img.shape[:2]  # Get height and width

            # Check if the image is grayscale or RGB
            if len(self.original_img.shape) == 2:  # Grayscale image
                image_format = QImage.Format.Format_Grayscale8

            else:  # RGB image
                image_format = QImage.Format.Format_RGB888

            # Resize the image while preserving aspect ratio
            aspect_ratio = width / height
            target_width = min(self.width(), int(self.height() * aspect_ratio))
            target_height = min(self.height(), int(self.width() / aspect_ratio))
            self.resized_img = cv2.resize(self.original_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            # Calculate the position to center the image
            x_offset = (self.width() - target_width) // 2
            y_offset = (self.height() - target_height) // 2

            # Convert image to QImage
            image = QImage(self.resized_img.data, self.resized_img.shape[1], self.resized_img.shape[0],
                           self.resized_img.strides[0], image_format)

            # Draw the image on the widget with the calculated offsets
            painter_img.drawImage(x_offset, y_offset, image)

            # Draw points if drawing is enabled
            if self.drawing_enabled:
                painter_img.setPen(QPen(QColor(Qt.GlobalColor.red), 4, Qt.PenStyle.SolidLine))
                for point in self.points:
                    # Draw X-shaped points
                    size = 6  # Size of X
                    painter_img.drawLine(point.x() - size, point.y() - size, point.x() + size, point.y() + size)
                    painter_img.drawLine(point.x() - size, point.y() + size, point.x() + size, point.y() - size)

            # Destroy the painter after use
            del painter_img  # This ensures proper cleanup

    def clear(self):
        """
        This method sets the `original_img` attribute to None, effectively clearing the currently displayed image.
        It then triggers an update of the display to reflect the change.

        Parameters:
            None

        Returns:
            None
        """
        self.original_img = None
        self.points = []
        self.repaint()

    def clear_points(self):
        self.points = []
        self.repaint()

    def mousePressEvent(self, event):
        if self.drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            # Add the clicked point
            self.points.append(event.pos())
            self.update()


    def enable_drawing(self):
        """
        Enable drawing for this instance.
        """
        self.drawing_enabled = True

    def disable_drawing(self):
        """
        Disable drawing for this instance.
        """
        self.drawing_enabled = False

    def get_drawing_points(self):
        """
        Returns a list of tuples representing the drawing coordinates.

        Returns:
            list: A list of tuples (x, y) representing the drawing positions.
        """
        return [(point.x(), point.y()) for point in self.points]  # Convert QPoint to tuple

# Example usage
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window1 = ImageViewport()
    window1.set_image("images\Screenshot 2024-04-21 231009.png")  # Replace "example.jpg" with your image path
    window1.enable_drawing()  # Enable drawing for this instance
    window1.show()

    window2 = ImageViewport()
    window2.set_image("images\Screenshot 2024-04-21 231009.png")  # Replace "example.jpg" with your image path
    window2.show()
    
    sys.exit(app.exec())
