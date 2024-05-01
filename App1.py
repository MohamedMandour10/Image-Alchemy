import cv2
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog
from PyQt6.QtGui import QIcon
import sys
import pyqtgraph as pg
from functools import partial
from src.Filters import Filter
from src.Noise import Noise
from src.Hybrid import Hybrid
from src.Edge_Detector import EdgeDetector
from src.imageViewPort import ImageViewport
from src.Thresholding import thresholding
from src.Decoding import Decoding
from src.Histogram import get_histograms

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.reset_buttons = None
        self.clear_buttons = None
        self.import_buttons = None
        self.ui_out_ports = None
        self.ui_view_ports = None
        self.out_ports = None
        self.input_ports = None
        self.ui = None
        self.original_img = None
        self.kernal_size = None
        self.hybrid = Hybrid()
        self.image_paths = [None for _ in range(6)]
        self.init_ui()


###################################################################################
#               UI binding functions and Helper Functions                         #
###################################################################################
        
    def init_ui(self):
        """
        Initialize the UI by loading the UI page, setting the window title, loading UI elements, and checking a specific UI element.
        """
        # Load the UI Page
        self.ui = uic.loadUi('UI\App1.ui', self)
        self.setWindowTitle("Image Processing ToolBox")
        self.setWindowIcon(QIcon("icons/image-layer-svgrepo-com.png"))
        self.load_ui_elements()
        self.connect_to_UI()
        self.ui.kernalSize_3.setChecked(True)
        self.ui.comboBox_2.setCurrentIndex(1)
        self.kernal_size = self.get_kernal_size()
        self.ui.medianFilter.clicked.connect(partial(self.apply_filter_noise, action_type="median_filter", class_name=Filter))
        self.ui.averageFilter.clicked.connect(partial(self.apply_filter_noise, action_type="average_filter", class_name=Filter))
        self.ui.gaussianFilter.clicked.connect(partial(self.apply_filter_noise, action_type="gaussian_filter", class_name=Filter))
        self.ui.saltPepperNoise.clicked.connect(partial(self.apply_filter_noise, action_type="slat_and_pepper", class_name=Noise))
        self.ui.uniformNoise.clicked.connect(partial(self.apply_filter_noise, action_type="uniform_noise", class_name=Noise))
        self.ui.gaussianNoise.clicked.connect(partial(self.apply_filter_noise, action_type="gaussian_noise", class_name=Noise))
        self.ui.hybridSlider1.valueChanged.connect(partial(self.apply_changes, class_type= Hybrid, action_type= "None", index=3))
        self.ui.hybridSlider2.valueChanged.connect(partial(self.apply_changes, class_type= Hybrid, action_type= "None", index=4))
        self.ui.hybridButton.clicked.connect(partial(self.apply_changes, class_type= Hybrid, action_type= "generate_hybrid", index=5))
        self.ui.comboBox_1.currentIndexChanged.connect(partial(self.onComboBoxChanged, isFirst = True))
        self.ui.comboBox_2.currentIndexChanged.connect(partial(self.onComboBoxChanged, isFirst = False))

    def load_ui_elements(self):
        """
        Load UI elements and set up event handlers.
        """
        # Initialize input and output port lists
        self.input_ports = []
        self.out_ports = []

        # Define lists of original UI view ports, output ports
        self.ui_view_ports = [self.ui.filterInput, self.ui.edgeInput,
                              self.ui.thresholdInput, self.ui.hybridIntput1, self.ui.hybridIntput2]

        self.ui_out_ports = [self.ui.filterOutput, self.ui.edgeOutput,
                             self.ui.thresholdOutput, self.ui.hybridOutput1, self.ui.hybridOutput2, self.ui.hybridOutput]

        # Create image viewports for input ports and bind browse_image function to the event
        self.input_ports.extend([
            self.create_image_viewport(self.ui_view_ports[i], lambda event, index=i: self.browse_image(event, index))
            for i in range(5)])

        # Create image viewports for output ports
        self.out_ports.extend(
            [self.create_image_viewport(self.ui_out_ports[i], mouse_double_click_event_handler=None) for i in range(6)])

        # Initialize import buttons
        self.import_buttons = [self.ui.importButton, self.ui.importButton_2,
                               self.ui.importButton_3, self.ui.importButton_hybrid1, self.ui.importButton_hybrid2]

        # Bind browse_image function to import buttons
        self.bind_buttons(self.import_buttons, self.browse_image)

        # Initialize clear buttons
        self.clear_buttons = [self.ui.clearButton, self.ui.clearButton_2,
                              self.ui.clearButton_3]

        # Bind clear function to clear buttons
        self.bind_buttons(self.clear_buttons, self.clear)

        # Initialize reset buttons
        self.reset_buttons = [self.ui.resetButton, self.ui.resetButton_2,
                              self.ui.resetButton_3]

        # Bind reset_image function to reset buttons
        self.bind_buttons(self.reset_buttons, self.reset_image)

        # Set range for frequency response of images
        self.ui.hybridSlider1.setRange(0, 255)
        self.ui.hybridSlider2.setRange(0, 255)

    def bind_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        if len(buttons) == 5:
            for i, button in enumerate(buttons):
                button.clicked.connect(lambda event, index=i: function(event, index))
        else:
            for i, button in enumerate(buttons):
                button.clicked.connect(lambda index=i: function(index))

    def connect_to_UI(self):
        connects = {
            self.ui.sobelEdge: (EdgeDetector, "sobel_detector", 1),
            self.ui.robertsEdge: (EdgeDetector, "roberts_detector", 1),
            self.ui.cannyEdge: (EdgeDetector, "canny_detector", 1),
            self.ui.prewittEdge: (EdgeDetector, "prewitt_detector", 1),
            self.ui.localThreshold: (thresholding, "local_thresholding", 2),
            self.ui.globalThreshold: (thresholding, "global_thresholding", 2),
            self.ui.equalizeButoon: (Decoding, "equalize", 2),
            self.ui.normalizeButton: (Decoding, "normalize", 2)
        }

        # Connect UI elements to the apply_changes method using the dictionary
        for ui_element, (class_type, action_type, index) in connects.items():
            ui_element.clicked.connect(partial(self.apply_changes, class_type=class_type, action_type=action_type, index=index))


    def get_kernal_size(self):
        if self.ui.kernalSize_3.isChecked():
            return 3
        else:
            return 5
        

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
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './', filter=file_filter)
        self.image_paths[index] = image_path

        # Check if the image path is valid and the index is within the range of input ports
        if image_path and 0 <= index < len(self.input_ports):
            # Check if the index is for the hybrid tab
            if index > 2:
                # Set the image for the last hybrid viewport
                input_port = self.input_ports[index]
                output_port = self.out_ports[index]
                input_port.set_image(image_path)
                output_port.set_image(image_path, grey_flag=True)
                if index == 3:
                    self.apply_changes(class_type= Hybrid, action_type= "low_pass" if self.ui.comboBox_1.currentIndex() == 0 else "high_pass", index=index)
                elif index == 4:
                    self.apply_changes(class_type= Hybrid, action_type= "low_pass" if self.ui.comboBox_2.currentIndex() == 0 else "high_pass", index=index)
            # Show the image on all viewports except the last hybrid viewport
            else:
                for idx, (input_port, output_port) in enumerate(zip(self.input_ports[:3], self.out_ports[:3])):
                    input_port.set_image(image_path)
                    output_port.set_image(image_path, grey_flag=True)

        # Generate histograms and distributions
        self.generate_hists_and_dists(index)


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
        Clear all the input and output ports.

        Args:
            index (int): The index of the port to clear.
        """
        for _, (input_port, output_port) in enumerate(zip(self.input_ports[:3], self.out_ports[:3])):
            input_port.clear()  # Clear the input port
            output_port.clear()  # Clear the output port
        self.clear_histographs()  # Clear the histographs


    def reset_image(self, index: int):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        if self.image_paths[index] is not None:
            self.input_ports[index].set_image(self.image_paths[index])
            self.out_ports[index].set_image(self.image_paths[index], grey_flag=True)


###################################################################################
#               Apply different image processing to the input image               #
###################################################################################

    def apply_changes(self, class_type, action_type: str, index: int): 
        """
        Apply changes to the input image using the specified class and action.

        Args:
        - class_type: The type of filter to apply to the image.
        - action_type: The action to perform on the filter.
        - index: The index of the input image to process.

        Returns:
        None
        """
        # Reset the image at the specified index
        if index < 3:
            self.reset_image(index)

        # Obtain the output port for the specified index
        output_port = self.out_ports[index]

        # Obtain the resized image from the output port
        img = output_port.resized_img

        # Check if the image is empty or None
        if (img is None or img.size == 0) and index != 5:
            print("Error: Empty or None image received.")
            return

        # Create an instance of the specified class type and apply the action
        try: 
            if index == 3:
                input_image = self.input_ports[index].resized_img
                action_type = "low_pass" if self.ui.comboBox_1.currentIndex() == 0 else "high_pass"
                action = self.hybrid
                action_method = getattr(action, action_type)
                processed_image = action_method(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), self.ui.hybridSlider1.value())
            elif index == 4:
                input_image = self.input_ports[index].resized_img
                action_type = "low_pass" if self.ui.comboBox_2.currentIndex() == 0 else "high_pass"
                action = self.hybrid
                action_method = getattr(action, action_type)
                processed_image = action_method(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), self.ui.hybridSlider2.value())
            else:
                action = class_type(img) if index < 5 else self.hybrid
                action_method = getattr(action, action_type)
                processed_image = action_method()

            # Update the original image in the output port and update the display
            output_port.original_img = processed_image
            output_port.update_display()
        except Exception as e:
            print(f"Error applying filter: {e}")


    def apply_filter_noise(self, action_type, class_name):
        """
        Apply median filter to the image and update the output port with the filtered image.
        """
        if class_name == Noise:
            self.reset_image(0)
        self.original_img = cv2.cvtColor(self.input_ports[0].resized_img.copy(), cv2.COLOR_BGR2GRAY)
        output_port = self.out_ports[0]
        img = output_port.resized_img
        if img is None or img.size == 0:
            print("Error: Empty or None image received.")
            return
        if class_name == Filter:
            action = class_name(img, self.kernal_size)
        else:
            action = class_name(img)
        try:
            filter_method = getattr(action, action_type)
            filtered_image = filter_method()
            output_port.original_img = filtered_image
            output_port.update_display()
        except Exception as e:
            print(f"Error applying median filter: {e}")


    def generate_hists_and_dists(self, index):
        """
        Generates histograms and distribution plots for the input image.
        """
        # Copy the original image
        
        self.original_img = self.input_ports[index].original_img
        
        # Compute histograms and distributions
        hists_and_dists = get_histograms(self.original_img)

        # Define plot widgets and corresponding data
        self.df_widgets = {
            'redDF_widget': (hists_and_dists['R'][0], hists_and_dists['R'][1], 'r'),
            'greenDF_widget': (hists_and_dists['G'][0], hists_and_dists['G'][1], 'g'),
            'blueDF_widget': (hists_and_dists['B'][0], hists_and_dists['B'][1], 'b'),
        }

        self.cdf_widgets = {
            'redCDF_widget': (hists_and_dists['R'][1], (255, 0, 0, 100), 'r'),
            'greenCDF_widget': (hists_and_dists['G'][1], (0, 255, 0, 100), 'g'),
            'blueCDF_widget': (hists_and_dists['B'][1], (0, 0, 255, 100), 'b'),
        }

        # Clear existing plot items
        self.clear_histographs()

        # Plot each widget
        for widget_name, (hist_data, bins_data, color) in self.df_widgets.items():
            # Set the background color to be transparent
            getattr(self.ui, widget_name).setBackground(None)

            # Plot histogram bars
            bar_plot = pg.BarGraphItem(x=bins_data, height=hist_data, width=1, pen=color)
            getattr(self.ui, widget_name).addItem(bar_plot)


        for widget_name, (data, brush_color, pen_color) in self.cdf_widgets.items():
            # Set the background color to be transparent
            getattr(self.ui, widget_name).setBackground(None)

            # Plot data
            plot_item = getattr(self.ui, widget_name).plot(data, pen=pen_color)
            plot_item.setFillLevel(0)
            plot_item.setBrush(pg.mkColor(brush_color))



    def clear_histographs(self):
        """
        Clears the histogram graphs for all plot widgets.
        """
        for df_widget, cdf_widget in zip(self.df_widgets.keys(), self.cdf_widgets.keys()):
            getattr(self.ui, df_widget).clear()
            getattr(self.ui, cdf_widget).clear()


    # Change between low pass and high pass filters in Hybrid
    def onComboBoxChanged(self, isFirst: bool):
        if isFirst:
            self.ui.comboBox_2.setCurrentIndex(1 if self.ui.comboBox_1.currentIndex() == 0 else 0)
        else: 
            self.ui.comboBox_1.setCurrentIndex(1 if self.ui.comboBox_2.currentIndex() == 0 else 0)   
        self.apply_changes(class_type= Hybrid, action_type= "low_pass" if self.ui.comboBox_1.currentIndex() == 0 else "high_pass", index=3)     
        self.apply_changes(class_type= Hybrid, action_type= "low_pass" if self.ui.comboBox_2.currentIndex() == 0 else "high_pass", index=4)     


def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
