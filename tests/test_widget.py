import logging

import numpy as np
import pytest

from napari_asegmenter._widget import (
    SegmenterWidget,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "shape, model_name, shared_memory",
    [
        ((100, 100), "StarDist", False),
        ((64, 64), "Cellpose", True),
        ((128, 128, 3), "SAM", True),
    ],
)
def test_segmenter_widget(
    make_napari_viewer, shape, model_name, shared_memory
):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.randint(0, 255, shape).astype(np.uint8))

    # create our widget, passing in the viewer
    widget = SegmenterWidget(viewer)
    widget._segmenter_combo.setCurrentText(model_name)
    widget._update_image_layers()
    widget._image_layer_combo.setCurrentIndex(0)
    widget._shared_memory_checkbox.setChecked(shared_memory)

    # call our widget method
    widget.segment()

    segmented_layers = [
        layer
        for layer in viewer.layers
        if layer.__class__.__name__ == "Labels"
        and layer.name.endswith("_segmented")
    ]

    assert segmented_layers[0].data.shape == shape[:2]

    widget.hideEvent(None)
