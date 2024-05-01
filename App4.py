from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon
import sys
import cv2
import numpy as np
from src.imageViewPort_exteneded import ImageViewport
from sklearn.cluster import AgglomerativeClustering
from src.Segmentation import  MeanShiftSegmentation, RegionGrowing, kmeans_segment_
from src.Threshold import Thresholding


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()

   
    def init_ui(self):
        """
        Initialize the UI by loading the UI page, setting the window title, loading UI elements, and checking a specific UI element.
        """
        # Load the UI Page
        self.ui = uic.loadUi('UI\App4.ui', self)
        self.setWindowTitle("Image Processing ToolBox")
        self.setWindowIcon(QIcon("icons/image-layer-svgrepo-com.png"))
        self.load_ui_elements()
        self.init_sliders()
        self.update_label_text()
        self.ui.clusters_comboBox.currentIndexChanged.connect(self.handle_clusters_combobox)
        self.ui.applyThreshold.clicked.connect(self.apply_thresholding)
        self.ui.applyCluster.clicked.connect(self.apply_clustering)
        self.ui.localRadio.setChecked(True)
        self.ui.localRadio.toggled.connect(self.radioToggled)
        self.ui.globalRadio.toggled.connect(self.radioToggled)
        self.ui.windowSlider.valueChanged.connect(self.update_label_text)
        self.ui.clustersSlider.valueChanged.connect(self.update_label_text)


    def load_ui_elements(self):
        """
        Load UI elements and set up event handlers.
        """
        # Initialize input and output port lists
        self.input_ports = []
        self.out_ports = []

        # Define lists of original UI view ports, output ports
        self.ui_view_ports = [self.ui.input1, self.ui.input2]

        self.ui_out_ports = [self.ui.output1, self.ui.output2]

        # Create image viewports for input ports and bind browse_image function to the event
        self.input_ports.extend([
            self.create_image_viewport(self.ui_view_ports[i], lambda event, index=i: self.browse_image(event, index))
            for i in range(2)])

        # Create image viewports for output ports
        self.out_ports.extend(
            [self.create_image_viewport(self.ui_out_ports[i], mouse_double_click_event_handler=None) for i in range(2)])
        
        self.out_ports[1].enable_drawing() 

        # Initialize import buttons
        self.import_buttons = [self.ui.importButton, self.ui.importButton2]

        # Bind browse_image function to import buttons
        self.bind_import_buttons(self.import_buttons, self.browse_image)

        # Initialize reset buttons
        self.reset_buttons = [self.ui.resetButton, self.ui.resetButton2]

        # Bind reset_image function to reset buttons
        self.bind_buttons(self.reset_buttons, self.reset_image)

        # Initialize reset buttons
        self.clear_buttons = [self.ui.clearButton, self.ui.clearButton2]

        # Call bind_buttons function
        self.bind_buttons(self.clear_buttons, self.clear_image)


    def bind_import_buttons(self, buttons, function):
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


    def bind_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        print(f"Binding buttons")
        for i, button in enumerate(buttons):
            print(f"Binding button {i}")
            button.clicked.connect(lambda index=i: function(index))


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
    

    def init_sliders(self):
        self.ui.clustersSlider.setRange(2, 10)
        self.ui.clustersSlider.setValue(2)

        self.ui.windowSlider.setRange(3, 20)
        self.ui.windowSlider.setValue(5)


    def update_label_text(self):
        """
        Updates the label text based on the current value of the sliders.

        This function is connected to the slider valueChanged signal,
        and is called whenever the value of a slider changes.
        It updates the text of the label next to the slider to display
        the current value of the slider.
        """

        # For Threshold
        window_size = self.ui.windowSlider.value()
        self.ui.window_val.setText(f"{window_size}")

        clusters_num = self.ui.clustersSlider.value()
        self.ui.cluster_val.setText(f"{clusters_num}")


    def handle_clusters_combobox(self):
        current_index = self.ui.clusters_comboBox.currentIndex()
        if current_index == 0 or current_index ==3:
            self.ui.clustersLabel.show()
            self.ui.clustersSlider.show()
            self.ui.cluster_val.show()
            self.ui.line_4.show()
        
        else:
            self.ui.clustersLabel.hide()
            self.ui.clustersSlider.hide()
            self.ui.cluster_val.hide()
            self.ui.line_4.hide()


    def radioToggled(self):
        if self.ui.localRadio.isChecked():
            self.ui.windowSizeLabel.show()
            self.ui.windowSlider.show()
            self.ui.window_val.show()
            self.ui.line_5.show()
        
        else:
            self.ui.windowSizeLabel.hide()
            self.ui.windowSlider.hide()
            self.ui.window_val.hide()
            self.ui.line_5.hide()

    def clear_image(self, index):
        """
        Clear the specifed input and output ports.

        Args:
            index (int): The index of the port to clear.
        """
        print(f"Clearing port {index}")
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


    def apply_thresholding(self):
        if self.ui.localRadio.isChecked():
            self.apply_local_thresholding()
        else:
            self.apply_global_thresholding()


    def apply_local_thresholding(self):
        selected_method = self.ui.threshold_comboBox.currentText()
        image = self.input_ports[0].original_img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholding = Thresholding(image=image)
        if selected_method == 'Optimal':
            output = thresholding.optimal_threshold_local(block_size=self.ui.windowSlider.value())
        elif selected_method == 'Otsu':
            output = thresholding.otsu_threshold_local()
        else:
            output = thresholding.spectral_threshold_local(block_size=self.ui.windowSlider.value())

        self.out_ports[0].original_img = output
        self.out_ports[0].update_display()


    def apply_global_thresholding(self):
        selected_method = self.ui.threshold_comboBox.currentText()
        image = self.input_ports[0].original_img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholding = Thresholding(image=image)
        if selected_method == 'Optimal':
            _ , output = thresholding.optimal_threshold_global()
        elif selected_method == 'Otsu':
            output = thresholding.otsu_threshold_global()
        else:
            output = thresholding.spectral_threshold_global()

        self.out_ports[0].original_img = output
        self.out_ports[0].update_display()


    def apply_clustering(self):
        selected_method = self.ui.clusters_comboBox.currentText()
        image = self.input_ports[1].original_img.copy()
        print("processing...")
        if selected_method == 'K-means':
            output = kmeans_segment_(image.copy(), self.ui.clustersSlider.value())
        elif selected_method == 'Region-Growing':
            segmentaion = RegionGrowing(image = image, seeds= self.out_ports[1].get_drawing_points())
            output = segmentaion.region_growing()
        elif selected_method == 'Mean-Shift':
            segmentaion = MeanShiftSegmentation(image = image)
            output = segmentaion.segment_image()
        else:
            output = self.apply_agglomeration(n_clusters = self.ui.clustersSlider.value())
            import time
            time.sleep(10)

        self.out_ports[1].original_img = output
        self.out_ports[1].update_display()
        print("Done")


    def apply_agglomeration(self, n_clusters=3):
        """
        Apply the agglomerative clustering algorithm to the image.

        Parameters:
            n_clusters (int): The number of clusters to create. Default is 3.

        Returns:
            numpy.ndarray: A segmented image with the same shape as the input image.
        """
        # Step 1: Read the image 
        image = cv2.imread(self.image_path)

        # Step 2: Convert color space from BGR to LUV
        image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        # Downsample the image to reduce memory usage (if necessary)
        downsample_factor = 0.2  # Adjust this factor as needed
        image_luv = image_luv[::int(1/downsample_factor), ::int(1/downsample_factor), :]

        # Step 3: Reshape the image into a feature matrix
        w, h, d =  tuple(image_luv.shape)
        assert d == 3
        image_array = np.reshape(image_luv, (w * h, d))

        # Step 4: Apply Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(image_array)

        # Step 5: Reshape the cluster labels to match the original image shape
        labels = clustering.labels_.reshape(w, h)

        # Step 6: Calculate mean intensity of each cluster
        segmented_image = np.zeros_like(image_luv, dtype=np.uint8)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_pixels = image_luv[labels == label]
            mean_intensity = np.mean(cluster_pixels, axis=0)
            segmented_image[labels == label] = mean_intensity.astype(np.uint8)

        return segmented_image


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
            output_port = self.out_ports[index]
            output_port.set_image(self.image_path, grey_flag=True)



def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()