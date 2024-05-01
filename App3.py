from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon
import sys
from src.imageViewPort import ImageViewport
from src.lambdaM import LambdaMin
from src.Harris import Harris
from src.MatchFeatures import MatchFeatures


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()

   
    def init_ui(self):
        """
        Initialize the UI by loading the UI page, setting the window title, loading UI elements, and checking a specific UI element.
        """
        # Load the UI Page
        self.ui = uic.loadUi('UI\App3.ui', self)
        self.setWindowTitle("Image Processing ToolBox")
        self.setWindowIcon(QIcon("icons/image-layer-svgrepo-com.png"))
        self.ui.applyButton1.clicked.connect(self.apply_operator)
        self.ui.clearButton.clicked.connect(self.clear)
        self.ui.applyButton2.clicked.connect(self.apply_match_features)
        self.ui.clearButton2.clicked.connect(self.clear)
        self.load_ui_elements()


    def load_ui_elements(self):
        """
        Load UI elements and set up event handlers.
        """
        # Initialize input and output port lists
        self.input_ports = []
        self.out_ports = []

        # Define lists of original UI view ports, output ports
        self.ui_view_ports = [self.ui.input1, self.ui.input2, self.ui.input3]

        self.ui_out_ports = [self.ui.output1, self.ui.output2]

        # Create image viewports for input ports and bind browse_image function to the event
        self.input_ports.extend([
            self.create_image_viewport(self.ui_view_ports[i], lambda event, index=i: self.browse_image(event, index))
            for i in range(3)])

        # Create image viewports for output ports
        self.out_ports.extend(
            [self.create_image_viewport(self.ui_out_ports[i], mouse_double_click_event_handler=None) for i in range(2)])

        # Initialize import buttons
        self.import_buttons = [self.ui.importButton, self.ui.importButton2]

        # Bind browse_image function to import buttons
        self.bind_buttons(self.import_buttons, self.browse_image)

        # Initialize reset buttons
        self.reset_buttons = [self.ui.resetButton, self.ui.resetButton2]

        # Bind reset_image function to reset buttons
        self.bind_buttons(self.reset_buttons, self.reset_image)


    def bind_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        for i, button in enumerate(buttons):
            button.clicked.connect(lambda event, index=i: function(event, index))

    def create_viewport(self, parent, viewport_class, mouse_double_click_event_handler=None):
        """
        Creates a viewport of the specified class and adds it to the specified parent widget.

        Args:
            parent: The parent widget to which the viewport will be added.
            viewport_class: The class of the viewport to be created.
            mouse_double_click_event_handler: The event handler function to be called when a mouse double-click event occurs (optional).

        Returns:
            The created viewport.

        """
        # Create a new instance of the viewport_class
        new_port = viewport_class(self)

        # Create a QVBoxLayout with parent as the parent widget
        layout = QVBoxLayout(parent)

        # Add the new_port to the layout
        layout.addWidget(new_port)

        # If a mouse_double_click_event_handler is provided, set it as the mouseDoubleClickEvent handler for new_port
        if mouse_double_click_event_handler:
            new_port.mouseDoubleClickEvent = mouse_double_click_event_handler

        # Return the new_port instance
        return new_port


    def create_image_viewport(self, parent, mouse_double_click_event_handler):
        """
        Creates an image viewport within the specified parent with the provided mouse double click event handler.
        """
        return self.create_viewport(parent, ImageViewport, mouse_double_click_event_handler)


    def clear(self, index: int):
        """
        Clear the specifed input and output ports.

        Args:
            index (int): The index of the port to clear.
        """

        self.input_ports[index].clear()
        self.out_ports[index].clear()


    def reset_image(self, index: int):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        self.input_ports[index].set_image(self.image_path)
        self.out_ports[index].set_image(self.image_path, grey_flag=True)


    def apply_operator(self):
        """
        This function applies the changes to the images based on the current index of the Hough Transform combobox.
        """
        current_index = self.ui.operators_comboBox.currentIndex()

        if current_index == 0:
            self.apply_harris()

        elif current_index == 1:
            self.apply_lambda()


    def apply_harris(self):
        input_image = self.input_ports[0].original_img.copy()
        output_port = self.out_ports[0]
        harris = Harris()
        output_img = harris.harris_corner_detector(input_image, k=0.04, threshold=0.01)
        output_port.original_img = output_img.copy() 
        output_port.update_display()  

    def apply_lambda(self):
        input_image = self.input_ports[0].original_img.copy()
        output_port = self.out_ports[0]
        lambda_min = LambdaMin()
        output_img = lambda_min.detect_corners(input_image, threshold=0.998, window=3)
        output_port.original_img = output_img.copy()  
        output_port.update_display()  

    def apply_match_features(self):
        current_index = self.ui.match_combo_box.currentIndex()

        if current_index == 0:
            self.apply_ssd()

        elif current_index == 1:
            self.apply_ncc()
    
    def apply_ssd(self):
        input_image1 = self.input_ports[1].original_img.copy()
        input_image2 = self.input_ports[2].original_img.copy()
        output_port = self.out_ports[1]
        match_features = MatchFeatures()
        output_img = match_features.match(input_image1, input_image2,method= 'ssd')
        output_port.original_img = output_img.copy()  
        output_port.update_display()  

    def apply_ncc(self):
        input_image1 = self.input_ports[1].original_img.copy()
        input_image2 = self.input_ports[2].original_img.copy()
        output_port = self.out_ports[1]
        match_features = MatchFeatures()
        output_img = match_features.match(input_image1, input_image2,method= 'ncc')
        output_port.original_img = output_img.copy()  
        output_port.update_display()  


    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

###################################################################################
#               Browse Image Function and Viewports controls                      #
###################################################################################
        

    def browse_image(self, event, index: int):
        """
        Browse for an image file and set it for the ImageViewport at the specified index.

        Args:
            event: The event that triggered the image browsing.
            index: The index of the ImageViewport to set the image for.
        """
        # Define the file filter for image selection
        file_filter = "Raw Data (*.png *.jpg *.jpeg *.jfif)"

        # Open a file dialog to select an image file
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './', filter=file_filter)

        # Check if the image path is valid and the index is within the range of input ports
        if self.image_path and 0 <= index < len(self.input_ports):

            # Set the image for the last hybrid viewport
            input_port = self.input_ports[index]
            input_port.set_image(self.image_path)
            if index == 0:
                output_port = self.out_ports[index]
                output_port.set_image(self.image_path, grey_flag=True)






def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()