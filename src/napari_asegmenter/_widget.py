from napari.viewer import Viewer
from napari_wsegmenter.core._segmenter_manager import SegmenterManager
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SegmenterWidget(QWidget):

    _segmenter_manager = SegmenterManager()

    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # Create layout
        layout = QVBoxLayout()

        # Image layer combo
        self._image_layer_label = QLabel("Image")
        self._image_layer_combo = QComboBox()
        layout.addWidget(self._image_layer_label)
        layout.addWidget(self._image_layer_combo)

        # Segmenter combo
        self._segmenter_label = QLabel("Segmenter")
        self._segmenter_combo = QComboBox()
        self._segmenter_combo.addItems(["StarDist", "Cellpose", "SAM"])
        layout.addWidget(self._segmenter_label)
        layout.addWidget(self._segmenter_combo)

        # Shared Memory checkbox
        self._shared_memory_checkbox = QCheckBox("Shared Memory")
        layout.addWidget(self._shared_memory_checkbox)

        # Run button
        self._run_button = QPushButton("Segment")
        self._run_button.clicked.connect(self.segment)
        layout.addWidget(self._run_button)

        self.setLayout(layout)

        # Populate image combo when viewer updates
        self._viewer.layers.events.inserted.connect(self._update_image_layers)
        self._viewer.layers.events.removed.connect(self._update_image_layers)
        self._update_image_layers()

    def _update_image_layers(self, event=None):
        self._image_layer_combo.clear()
        image_layers = [
            layer.name
            for layer in self._viewer.layers
            if layer.__class__.__name__ == "Image"
        ]
        self._image_layer_combo.addItems(image_layers)

    def segment(self):
        image_name = self._image_layer_combo.currentText()
        segmenter = self._segmenter_combo.currentText().lower()

        if image_name not in self._viewer.layers:
            return

        print(f"Running '{segmenter}' on image layer '{image_name}'")

        image_layer = self._viewer.layers[image_name]

        segmentation = self._segmenter_manager.perform_segmentation(
            image_layer.data,
            segmenter,
            self._shared_memory_checkbox.isChecked(),
        )

        name = image_layer.name + "_segmented"
        if name in self._viewer.layers:
            self._viewer.layers[name].data = segmentation
        else:
            self._viewer.add_labels(segmentation, name=name)

    def hideEvent(self, a0):
        self._segmenter_manager.exit()
        super().hideEvent(a0)
